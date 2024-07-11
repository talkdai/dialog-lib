import warnings
from operator import itemgetter

from langchain.schema import format_document
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory.chat_memory import BaseChatMemory
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.conversation.memory import ConversationBufferMemory

from dialog_lib.db import get_session
from dialog_lib.db.memory import CustomPostgresChatMessageHistory, get_memory_instance
from dialog_lib.embeddings.retrievers import DialogRetriever


class AbstractLLM:
    def __init__(
        self,
        config,
        session_id=None,
        parent_session_id=None,
        dataset=None,
        llm_api_key=None,
        dbsession=get_session,
    ):
        """
        :param config: Configuration dictionary

        The constructor of the AbstractLLM class allows users to pass
        a configuration dictionary to the LLM. This configuration dictionary
        can be used to configure the LLM temperature, prompt and other
        necessities.
        """
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")

        self.config = config
        self.prompt = None
        self.session_id = None
        self.relevant_contents = None
        if session_id:
            self.session_id = (
                session_id if dataset is None else f"{dataset}_{session_id}"
            )
        self.dataset = dataset
        self.llm_api_key = self.config.get("llm_api_key", llm_api_key)
        self.parent_session_id = parent_session_id
        self.dbsession = dbsession

    @property
    def memory(self) -> BaseChatMemory:
        """
        Returns the memory instance
        """
        raise NotImplementedError("Memory must be implemented")

    @property
    def llm(self) -> LLMChain:
        """
        Returns the LLM instance. If a memory instance is provided,
        the LLM instance should be initialized with the memory instance.

        If no memory instance is provided, the LLM instance should be
        initialized without a memory instance.
        """
        raise NotImplementedError("LLM must be implemented")

    def preprocess(self, input: str) -> str:
        """
        Function that pre-process the LLM input, enabling users
        to modify the input before it is processed by the LLM.

        This can be used to add context or prefixes to the LLM.
        """
        return input

    def generate_prompt(self, text: str) -> str:
        """
        Function that generates the prompt using PromptTemplate for the LLM.
        """
        return text

    def postprocess(self, output: str) -> str:
        """
        Function that post-process the LLM output, enabling users
        to modify the output before it is returned to the user.
        """
        return output

    def process(self, input: str):
        """
        Function that encapsulates the pre-processing, processing and post-processing
        of the LLM.
        """
        processed_input = self.preprocess(input)
        self.generate_prompt(processed_input)
        output = self.llm.invoke(
            {
                "user_message": processed_input,
            }
        )
        processed_output = self.postprocess(output)
        return processed_output

    @property
    def messages(self):
        """
        Returns the messages from the memory instance
        """
        return self.memory.messages


class AbstractLCEL(AbstractLLM):
    def __init__(self, *args, **kwargs):
        kwargs["config"] = kwargs.get("config", {})
        self.memory_instance = kwargs.pop("memory", None)
        self.llm_api_key = kwargs
        self.prompt_content = kwargs.pop("prompt", None)
        self.chat_model = kwargs.pop("model_class")
        self.embedding_llm = kwargs.pop("embedding_llm")
        self.cosine_similarity_threshold = kwargs.pop("cosine_similarity_threshold", 0.3)
        self.top_k = kwargs.pop("top_k", 3)
        super().__init__(*args, **kwargs)

    @property
    def document_prompt(self):
        return PromptTemplate.from_template(template="{page_content}")

    @property
    def retriever(self):
        with self.dbsession() as session:
            return DialogRetriever(
                session=session,
                embedding_llm=self.embedding_llm,
                threshold=self.cosine_similarity_threshold,
                top_k=self.top_k
            )

    @property
    def model(self):
        return self.chat_model

    def combine_docs(self, docs, document_separator="\n\n"):
        """
        This is the default combine_documents function that returns the documents as is.
        We use the default combine_docs function from Langchain.
        """
        doc_strings = [format_document(doc, self.document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    @property
    def retriever_chain(self):
        """
        builds and returns the retriever chain for the LCEL
        """
        return (
            itemgetter("input") | self.retriever
        ).with_config({"run_name": "RetrieverChain"})

    @property
    def fallback_chain(self):
        """
        builds and returns the fallback message chain for the LCEL
        """
        fallback_prompt = ChatPromptTemplate.from_messages(
            [
                ("ai", self.config.get("prompt").get("fallback_not_found_relevant_contents"))
            ]
        )

        return (
            fallback_prompt | RunnableLambda(lambda x: x.messages[-1])
        )

    @property
    def answer_chain(self):
        """
        builds and returns the answer chain for the LCEL
        """
        return (
            RunnableParallel(
                {
                    "context": itemgetter("relevant_contents") | RunnableLambda(self.combine_docs),
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
            ).with_config({"run_name": "GetContext"})
            | self.prompt
            | self.model
        ).with_config({"run_name": "AnswerChain"})

    @property
    def answer_runnable(self):
        return RunnableWithMessageHistory(
            self.answer_chain,
            self.get_session_history,
            input_messages_key='input',
            history_messages_key="chat_history"
        ).with_config({"run_name": "AnswerRunnableWithHistory"})

    @property
    def memory(self):
        with self.dbsession() as session:
            return get_memory_instance(
                session_id=self.session_id,
                sqlalchemy_session=session,
                database_url=self.config.get("database_url")
            )

    def get_session_history(self, something):
        with self.dbsession() as session:
            return CustomPostgresChatMessageHistory(
                connection_string=self.config.get("database_url"),
                session_id=self.session_id,
                parent_session_id=self.parent_session_id,
                table_name="chat_messages",
                dbsession=session,
            )

    def chain_router(self, input):
        return self.answer_runnable if len(input["relevant_contents"]) > 0 else self.fallback_chain

    @property
    def main_chain(self):
        return (
            (
                RunnableParallel(
                    {
                        "relevant_contents": self.retriever_chain,
                        "input": itemgetter("input")
                    }
                ).with_config({"run_name": "GetRelevantContext"}) | RunnableLambda(self.chain_router).with_config(
                    {"run_name": "ChainRouter"}
                )
            )
        )

    def process(self, input: str):
        """
        Function that encapsulates the pre-processing, processing and post-processing
        of the LLM.
        """
        processed_input = self.preprocess(input)
        self.generate_prompt(processed_input)
        output = self.main_chain.invoke(
            {
                "input": processed_input,
            },
            {"configurable": {
                "session_id": self.session_id,
            }}
        )
        processed_output = self.postprocess(output)
        return processed_output

    def invoke(self, input: dict):
        """
        Function that invokes the LLM with the given input.
        """
        return self.process(input)

    def generate_prompt(self, input_text):
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "What can I help you with today?"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", "Here is some context for the user request: {context}"),
                ("human", input_text),
            ]
        )

    def postprocess(self, output):
        return output.content


class AbstractRAG(AbstractLLM):
    relevant_contents = []

    def process(self, input: str):
        """
        Function that encapsulates the pre-processing, processing and post-processing
        of the LLM.
        """
        processed_input = self.preprocess(input)
        self.generate_prompt(processed_input)
        if len(self.relevant_contents) == 0 and self.config.get("prompt").get(
            "fallback_not_found_relevant_contents"
        ):
            return {
                "text": self.config.get("prompt").get(
                    "fallback_not_found_relevant_contents"
                )
            }
        output = self.llm.invoke(
            {
                "user_message": processed_input,
            }
        )
        processed_output = self.postprocess(output)
        return processed_output


class AbstractDialog(AbstractLLM):
    def __init__(self, *args, **kwargs):
        warnings.filterwarnings("default", category=DeprecationWarning)
        warnings.warn(
            (
                "AbstractDialog will be deprecated in release 0.2 due to the creation of Langchain's LCEL. ",
                "Please use AbstractLCEL instead."
            ), DeprecationWarning, stacklevel=3
        )
        kwargs["config"] = kwargs.get("config", {})

        self.memory_instance = kwargs.pop("memory", None)
        self.llm_api_key = kwargs.pop("openai_api_key", None)
        self.prompt_content = kwargs.pop("prompt", None)
        self.chat_model = kwargs.pop("model_class")
        super().__init__(*args, **kwargs)

    def generate_prompt(self, input_text):
        initial_prompt = [
            ("system", self.prompt_content + "\n\n{chat_history}"),
            ("human", input_text)
        ]

        self.prompt = ChatPromptTemplate.from_messages(initial_prompt)
        return input_text

    @property
    def memory(self):
        return self.memory_instance

    @property
    def llm(self):
        chain_settings = dict(
            llm=self.chat_model,
            prompt=self.prompt
        )

        if self.memory:
            if isinstance(self.memory, ConversationBufferMemory):
                chain_settings["memory"] = self.memory
            else:
                buffer_config = {
                    "chat_memory": self.memory,
                    "memory_key": "chat_history",
                    "return_messages": True,
                    "k": self.config.get("memory_size", 5)
                }
                chain_settings["memory"] = ConversationBufferWindowMemory(
                    **buffer_config
                )
        return LLMChain(
            **chain_settings
        )


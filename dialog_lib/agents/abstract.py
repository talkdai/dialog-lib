from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dialog_lib.db.memory import CustomPostgresChatMessageHistory, get_memory_instance


class AbstractLLM:
    def __init__(
        self,
        config,
        session_id=None,
        parent_session_id=None,
        dataset=None,
        llm_api_key=None,
        dbsession=None,
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


class AbstractLcelClass(AbstractLLM):


    @property
    def chat_model(self):
        """
        builds and returns the chat model for the LCEL
        """
        raise NotImplementedError("Chat model must be implemented")

    @property
    def context_dict(self):
        """
        builds and returns the context dictionary for the LCEL
        """
        raise NotImplementedError("Context dictionary must be implemented")

    @property
    def chain(self):
        """
        builds and returns the chain for the LCEL
        """
        return (
            self.context_dict | self.prompt | self.chat_model
        )

    @property
    def get_memory_instance(self):
        return get_memory_instance(
            session_id=self.session_id,
            sqlalchemy_session=self.dbsession,
            database_url=self.config.get("database_url")
        )

    @property
    def runnable(self):
        RunnableWithMessageHistory(
            self.chain,
            self.get_memory_instance,
            input_messages_key='input',
            history_messages_key="chat_history"
        )

    def process(self, input: str):
        """
        Function that encapsulates the pre-processing, processing and post-processing
        of the LLM.
        """
        processed_input = self.preprocess(input)
        self.generate_prompt(processed_input)
        output = self.runnable.invoke(
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
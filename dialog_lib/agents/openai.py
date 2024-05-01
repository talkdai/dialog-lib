from .abstract import AbstractLLM
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory

from langchain.chains.llm import LLMChain
from langchain_openai.chat_models.base import ChatOpenAI


class DialogOpenAI(AbstractLLM):
    def __init__(self, *args, **kwargs):
        model = kwargs.pop("model", "gpt-3.5-turbo")
        temperature = kwargs.pop("temperature", 0.1)
        kwargs["config"] = kwargs.get("config", {})

        self.memory_instance = kwargs.pop("memory", None)
        self.llm_api_key = kwargs.pop("openai_api_key", None)
        self.prompt_content = kwargs.pop("prompt", None)

        super().__init__(*args, **kwargs)

        self.chat_model = ChatOpenAI(
            openai_api_key=self.llm_api_key,
            model=model,
            temperature=temperature
        )

    def generate_prompt(self, input_text):
        self.prompt = ChatPromptTemplate.from_messages([
            ("ai", self.prompt_content),
            ("human", input_text)
        ])
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

    def postprocess(self, output):
        return output.get("text")
import os
from .abstract import AbstractDialog, AbstractLCEL
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models.base import ChatOpenAI
from dialog_lib.embeddings.retrievers import DialogRetriever


class DialogOpenAI(AbstractDialog):
    def __init__(self, *args, **kwargs):
        model = kwargs.pop("model", "gpt-3.5-turbo")
        temperature = kwargs.pop("temperature", 0.1)
        kwargs["model_class"] = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=kwargs.get("llm_api_key"),
        )
        super().__init__(*args, **kwargs)

    def postprocess(self, output):
        return output.get("text")


class DialogLCELOpenAI(AbstractLCEL):
    def __init__(self, *args, **kwargs):
        self.openai_api_key = kwargs.get("llm_api_key") or os.environ.get("OPENAI_API_KEY")
        kwargs["model_class"] = ChatOpenAI(
            model=kwargs.pop("model"),
            temperature=kwargs.pop("temperature"),
            openai_api_key=self.openai_api_key,
        )
        kwargs["embedding_llm"] = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        super().__init__(*args, **kwargs)

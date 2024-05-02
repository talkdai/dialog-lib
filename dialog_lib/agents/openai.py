from .abstract import AbstractDialog
from langchain_openai.chat_models.base import ChatOpenAI


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
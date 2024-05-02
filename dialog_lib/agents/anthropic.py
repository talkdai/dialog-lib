from .abstract import AbstractDialog
from langchain_anthropic import ChatAnthropic


class DialogAnthropic(AbstractDialog):
    def __init__(self, *args, **kwargs):
        model = kwargs.pop("model", "claude-3-opus-20240229")
        temperature = kwargs.pop("temperature", 0)
        kwargs["model_class"] = ChatAnthropic(
            model=model,
            temperature=temperature,
            anthropic_api_key=kwargs.get("llm_api_key"),
            max_tokens=1536
        )
        super().__init__(*args, **kwargs)

    def postprocess(self, output):
        return output.get("text")
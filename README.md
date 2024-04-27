# Dialog Library

Welcome to our project! Here is all the information you need for you to start using dialog-lib as you desire.

Dialog-lib is the base library for the  project, it allows users to setup their own LLMs using a structured class, enabling coherent callings across all instances.
The main purpose of this project is removing the main difficulties

## Integrations

This is a standalone project, you can use it as a wrapper for LLM instances, right now we have the following LLMs integrated:

- [X] OpenAIs GPT Models (both 3.5 and 4) - available in the server's original repo, being migrated to this one
- [ ] Azure OpenAI
- [ ] Mistral
- [ ] Bedrock
- [ ] DataBricks  - In progress
- [ ] MLFlow
- [ ] Hugging Faces - In progress

## How to use

Right now, this repository offers just a single Abstract LLM class inside our agents module and some PostgreSQL memory helpers, we are moving our abstractions to make it easier to implement any LLM you want, giving you access to vector stores and memory instances.

## Future structure

The desired future of our abstraction is something in the lines of the code below:

```python
from dialog_lib.agents import OpenAIAgent

agent_instance = OpenAIAgent(model="gpt3.5-turbo", temperature=0.1, prompt="Be a friendly AI", memory_type="ram")

agent_instance.process("Hello There!")

print(agent_instance.messages) # Get's all the messages saved
```

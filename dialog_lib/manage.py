import click
import os
from dialog_lib.agents import DialogOpenAI, DialogAnthropic
from dialog_lib.memory import generate_local_memory_instance

@click.group()
def cli():
    pass


def run_llm(agent, instance_name, memory, debug=False):
    click.echo("## Welcome to talkd.ai/dialog-lib CLI")
    click.echo("## Initialized the LLM process")

    while True:
        input_text = input("You: ")

        if input_text == "q":
            break

        output_text = agent.process(input_text)
        print(f"{instance_name}: {output_text}")

    if debug:
        click.echo(memory.chat_memory.messages)



@cli.command()
@click.option("--model", default="gpt-3.5-turbo", help="The model to use")
@click.option("--temperature", default=0.1, help="The temperature for generating responses")
@click.option("--llm-api-key", default=os.environ.get("OPENAI_API_KEY"), help="The OpenAI API key")
@click.option("--prompt", default="You are a bot called Sara. Be nice to other human beings.", help="The prompt for the dialog")
@click.option("--debug", default=False, help="Prints the memory after the dialog ends", is_flag=True)
def openai(model, temperature, llm_api_key, prompt, debug):
    memory = generate_local_memory_instance()
    dialog = DialogOpenAI(
        model=model,
        temperature=temperature,
        llm_api_key=llm_api_key,
        prompt=prompt,
        memory=memory,
    )
    run_llm(dialog, "ChatGPT", memory, debug=debug)


@cli.command()
@click.option("--model", default="claude-3-opus-20240229", help="The model to use")
@click.option("--temperature", default=0, help="The temperature for generating responses")
@click.option("--llm-api-key", default=os.environ.get("ANTHROPIC_API_KEY", None), help="The Anthropic API key", required=True)
@click.option("--prompt", default="You are a bot called Sara. Be nice to other human beings.", help="The prompt for the dialog")
@click.option("--debug", default=False, help="Prints the memory after the dialog ends", is_flag=True)
def anthropic(model, temperature, llm_api_key, prompt, debug):
    memory = generate_local_memory_instance()
    dialog = DialogAnthropic(
        model=model,
        temperature=temperature,
        llm_api_key=llm_api_key,
        prompt=prompt,
        memory=memory,
    )
    run_llm(dialog, "Anthropic", memory, debug=debug)


def main():
    cli()

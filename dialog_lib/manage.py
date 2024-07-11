import os
import click
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from dialog_lib.loaders.csv import load_csv as csv_loader
from dialog_lib.loaders.gsheets import load_google_sheets as gsheets_loader
from dialog_lib.agents import DialogOpenAI, DialogAnthropic
from dialog_lib.memory import generate_local_memory_instance

from langchain_openai import OpenAIEmbeddings

@click.group()
def cli():
    pass

def get_llm_key(env=None):
    if env is not None:
        return os.environ.get(env)

    return os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY")

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
@click.option("--llm-api-key", default=get_llm_key(), help="The OpenAI API key")
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
@click.option("--llm-api-key", default=get_llm_key(), help="The Anthropic API key", required=True)
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

@cli.command()
@click.option("--database-url", default=os.environ.get("DATABASE_URL"), help="The postgres database URL")
@click.option("--llm-api-key", default=get_llm_key(), help="The LLM API key", required=True)
@click.option("--file", help="The CSV file to load the data from", required=True)
def load_csv(database_url, llm_api_key, file):
    engine = create_engine(database_url)
    dbsession = Session(engine.connect())
    with Session(engine.connect()) as session:
        csv_loader(
            file_path=file,
            dbsession=session,
            embedding_llm_model="openai",
            embedding_llm_api_key=llm_api_key
        )
    click.echo("## Loaded the CSV file to the database")

@cli.command()
@click.option("--spreadsheet-url", help="The Google Sheets URL", required=True)
@click.option("--sheet-name", help="The Google Sheets sheet name", required=True)
@click.option(
    "--credentials-path",
    help="The complete path for the Service Account credentials.json file, i.e.: /user/Talkd/Downloads/credentials.json",
    required=True
)
@click.option("--database-url", default=os.environ.get("DATABASE_URL"), help="The postgres database URL")
@click.option("--llm-api-key", default=get_llm_key(), help="The OpenAI API key")
def load_google_sheets(spreadsheet_url, sheet_name, credentials_path, database_url, llm_api_key):
    engine = create_engine(database_url)
    dbsession = Session(engine.connect())
    gsheets_loader(
        credentials_path=credentials_path,
        spreadsheet_url=spreadsheet_url,
        sheet_name=sheet_name,
        dbsession=dbsession,
        embeddings_model_instance=OpenAIEmbeddings(openai_api_key=llm_api_key)
    )

def main():
    cli()

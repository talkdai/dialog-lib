import click

@click.group()
def cli():
    pass

@cli.command()
def initllm():
    click.echo('Initialized the LLM process')

def main():
    cli()
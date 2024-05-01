import os

from sqlalchemy import create_engine
from dialog_lib.agents import DialogOpenAI
from dialog_lib.memory import generate_memory_instance


database_url = "postgresql://talkdai:talkdai@db:5432/test_talkdai"

engine = create_engine(database_url)

dbsession = engine.connect()


memory = generate_memory_instance(
    session_id="test_session",
    dbsession=dbsession,
    database_url=database_url,
)

agent = DialogOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
    llm_api_key=os.environ.get("OPENAI_API_KEY"),
    prompt="You are a bot called Sara. Be nice to other human beings.",
    memory=memory,
)

while True:
    input_text = input("You: ")
    output_text = agent.process(input_text)
    print(f"Sara: {output_text}")

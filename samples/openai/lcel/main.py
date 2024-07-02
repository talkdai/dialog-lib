import os
import logging
from uuid import uuid4
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from dialog_lib.agents.openai import DialogLCELOpenAI

logging.getLogger().setLevel(logging.ERROR)

database_url = "postgresql://talkdai:talkdai@db:5432/test_talkdai"

engine = create_engine(database_url)

dbsession = Session(engine)


agent = DialogLCELOpenAI(
    model="gpt-4o",
    temperature=0.1,
    llm_api_key=os.environ.get("OPENAI_API_KEY"),
    prompt="You are a bot called Sara. Be nice to other human beings.",
    dbsession=dbsession,
    config={
        "database_url": database_url,
        "prompt": {
            "fallback_not_found_relevant_contents": "I'm sorry, I don't have an answer for that. Can I help you with something else?",
        }
    },
    session_id=str(uuid4())
)

while True:
    input_text = input("You: ")
    output_text = agent.process(input_text)
    print(f"Sara: {output_text}")

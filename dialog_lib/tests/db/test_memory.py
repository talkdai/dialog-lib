import os
import sqlalchemy

from dialog_lib.db import (
    generate_memory_instance, add_user_message_to_message_history, get_messages,
    Chat, ChatMessages, CompanyContent
)
from dialog_lib.db.memory import CustomPostgresChatMessageHistory

from langchain_core.messages import HumanMessage


def test_models_were_created(db_engine):
    assert True == sqlalchemy.inspect(db_engine).has_table(Chat.__tablename__)
    assert True == sqlalchemy.inspect(db_engine).has_table(ChatMessages.__tablename__)
    assert True == sqlalchemy.inspect(db_engine).has_table(CompanyContent.__tablename__)

def test_generate_memory_instance(db_session):
    memory = generate_memory_instance(
        "test_session_id",
        database_url=os.environ.get('DATABASE_URL'),
        dbsession=db_session,
    )
    assert isinstance(memory, CustomPostgresChatMessageHistory)

def test_add_user_message_to_memory_instance_and_db(db_session):
    memory = add_user_message_to_message_history(
        "test_session_id",
        "test_message",
        dbsession=db_session,
        database_url=os.environ.get('DATABASE_URL'),
    )
    assert isinstance(memory.messages[0], HumanMessage)
    assert memory.messages[0].content == "test_message"

def test_get_messages(db_session):
    add_user_message_to_message_history(
        "test_session_id",
        "test_message",
        dbsession=db_session,
        database_url=os.environ.get('DATABASE_URL'),
    )
    messages = get_messages(
        "test_session_id",
        dbsession=db_session,
        database_url=os.environ.get('DATABASE_URL'),
    )
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "test_message"
import psycopg
from .session import get_session
from langchain_postgres import PostgresChatMessageHistory
from langchain.schema.messages import BaseMessage, _message_to_dict

from .models import Chat, ChatMessages


class CustomPostgresChatMessageHistory(PostgresChatMessageHistory):
    """
    Custom chat message history for LLM
    """

    def __init__(
        self,
        *args,
        parent_session_id=None,
        dbsession=get_session,
        chats_model=Chat,
        chat_messages_model=ChatMessages,
        ssl_mode=None,
        **kwargs,
    ):
        self.parent_session_id = parent_session_id
        self.dbsession = dbsession
        self.chats_model = chats_model
        self.chat_messages_model = chat_messages_model
        self._connection = psycopg.connect(
            kwargs.pop("connection_string"), sslmode=ssl_mode
        )
        self._session_id = kwargs.pop("session_id")
        self._table_name = kwargs.pop("table_name")


    def create_tables(self) -> None:
        """
        create table if it does not exist
        add a new column for timestamp
        """
        create_table_query = f"""CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            message JSONB NOT NULL,
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );"""
        self.cursor.execute(create_table_query)
        self.connection.commit()

    def add_tags(self, tags: str) -> None:
        """Add tags for a given session_id/uuid on chats table"""
        with self.dbsession() as session:
            session.query(self.chats_model).where(
                self.chats_model.session_id == self._session_id
            ).update({getattr(self.chats_model, "tags"): tags})

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in PostgreSQL"""
        message = self.chat_messages_model(
            session_id=self._session_id, message=_message_to_dict(message)
        )
        if self.parent_session_id:
            message.parent = self.parent_session_id
        self.dbsession.add(message)
        self.dbsession.commit()


def generate_memory_instance(
    session_id,
    parent_session_id=None,
    dbsession=get_session,
    database_url=None,
    chats_model=Chat,
    chat_messages_model=ChatMessages,
):
    """
    Generate a memory instance for a given session_id
    """

    return CustomPostgresChatMessageHistory(
        connection_string=database_url,
        session_id=session_id,
        parent_session_id=parent_session_id,
        table_name="chat_messages",
        dbsession=dbsession,
        chats_model=chats_model,
        chat_messages_model=chat_messages_model,
    )


def add_user_message_to_message_history(
    session_id, message, memory=None, dbsession=get_session(), database_url=None
):
    """
    Add a user message to the message history and returns the updated
    memory instance
    """
    if not memory:
        memory = generate_memory_instance(
            session_id, dbsession=dbsession, database_url=database_url
        )

    memory.add_user_message(message)
    return memory


def get_messages(session_id, dbsession=get_session(), database_url=None):
    """
    Get all messages for a given session_id
    """
    memory = generate_memory_instance(
        session_id, dbsession=dbsession, database_url=database_url
    )
    return memory.messages

def get_memory_instance(session_id, sqlalchemy_session, database_url):
    return generate_memory_instance(
        session_id=session_id,
        dbsession=sqlalchemy_session,
        database_url=database_url
    )
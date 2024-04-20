from langchain.memory import PostgresChatMessageHistory
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
        dbsession=None,
        chat_messages_model=ChatMessages,
        **kwargs,
    ):
        self.parent_session_id = parent_session_id
        self.dbsession = dbsession
        self.chat_messages_model = chat_messages_model
        super().__init__(*args, **kwargs)

    def _create_table_if_not_exists(self) -> None:
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
        self.dbsession.query(Chat).where(Chat.session_id == self.session_id).update(
            {Chat.tags: tags}
        )
        self.dbsession.commit()

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in PostgreSQL"""
        message = ChatMessages(
            session_id=self.session_id, message=_message_to_dict(message)
        )
        if self.parent_session_id:
            message.parent = self.parent_session_id
        self.dbsession.add(message)
        self.dbsession.commit()


def generate_memory_instance(
    session_id, parent_session_id=None, dbsession=None, database_url=None
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
    )


def add_user_message_to_message_history(
    session_id, message, memory=None, dbsession=None, database_url=None
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


def get_messages(session_id, dbsession=None, database_url=None):
    """
    Get all messages for a given session_id
    """
    memory = generate_memory_instance(
        session_id, dbsession=dbsession, database_url=database_url
    )
    return memory.messages

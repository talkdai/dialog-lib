import psycopg

from typing import List
from psycopg import sql

from .models import Chat, ChatMessages
from .session import get_session, get_async_session, get_async_psycopg_connection

from langchain_postgres import PostgresChatMessageHistory
from langchain.schema.messages import BaseMessage, _message_to_dict


class CustomPostgresChatMessageHistory(PostgresChatMessageHistory):
    """
    Custom chat message history for LLM
    """

    def __init__(
        self,
        *args,
        parent_session_id=None,
        dbsession=get_session,
        async_dbsession=get_async_session,
        chats_model=Chat,
        chat_messages_model=ChatMessages,
        ssl_mode=None,
        **kwargs,
    ):
        self.parent_session_id = parent_session_id
        self.dbsession = dbsession
        self.async_dbsession = async_dbsession
        self.chats_model = chats_model
        self.chat_messages_model = chat_messages_model
        self._connection = psycopg.connect(
            kwargs.pop("connection_string"), sslmode=ssl_mode
        )
        self._async_connection = None  # Will be initialized when needed
        self._session_id = kwargs.pop("session_id")
        self._table_name = kwargs.pop("table_name", chat_messages_model.__tablename__)

        self.cursor = self._connection.cursor()

    async def _initialize_async_connection(self):
        if self._async_connection is None:
            self._async_connection = await get_async_psycopg_connection()
        return self._async_connection

    def _create_tables_queries(self, table_name):
        index_name = f"idx_{table_name}_session_id"
        return [
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message JSONB NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );"""
            ).format(table_name=sql.Identifier(table_name)),
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} (session_id);
                """
            ).format(
                index_name=sql.Identifier(index_name),
                table_name=sql.Identifier(table_name)
            )
        ]

    def _get_messages_query(self, table_name):
        return [
            sql.SQL(
                """
                SELECT message FROM {table_name} WHERE session_id = {session_id};
                """
            ).format(
                table_name=sql.Identifier(table_name),
                session_id=sql.Literal(self._session_id)
            )
        ]

    def create_tables(self) -> None:
        """
        Create table if it does not exist
        Add a new column for timestamp
        """
        create_table_queries = self._create_tables_queries(self._table_name)
        for query in create_table_queries:
            self.cursor.execute(query)
        self._connection.commit()

    async def acreate_tables(self) -> None:
        """
        Asynchronously create tables.
        """
        create_table_queries = self._create_tables_queries(self._table_name)
        async_conn = await self._initialize_async_connection()
        async with async_conn.cursor() as cursor:
            for query in create_table_queries:
                await cursor.execute(query)
            await async_conn.commit()

    def get_messages(self):
        """
        Retrieve messages synchronously.
        """
        get_messages_query = self._get_messages_query(self._table_name)
        for query in get_messages_query:
            self.cursor.execute(query)
        return self.cursor.fetchall()

    async def aget_messages(self):
        """
        Retrieve messages asynchronously.
        """
        get_messages_query = self._get_messages_query(self._table_name)
        async_conn = await self._initialize_async_connection()
        async with async_conn.cursor() as cursor:
            for query in get_messages_query:
                await cursor.execute(query)
            return await cursor.fetchall()

    def add_tags(self, tags: str) -> None:
        """
        Add tags for a given session_id/uuid on chats table.
        """
        with self.dbsession() as session:
            session.query(self.chats_model).where(
                self.chats_model.session_id == self._session_id
            ).update({getattr(self.chats_model, "tags"): tags})
            session.commit()

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """
        Add messages to the record in PostgreSQL.
        """
        for message in messages:
            self.add_message(message)

    def add_message(self, message: BaseMessage) -> None:
        """
        Append the message to the record in PostgreSQL.
        """
        message = self.chat_messages_model(
            session_id=self._session_id, message=_message_to_dict(message)
        )
        if self.parent_session_id:
            message.parent = self.parent_session_id
        self.dbsession.add(message)
        self.dbsession.commit()

    async def aadd_messages(self, messages: List[BaseMessage]) -> None:
        """
        Asynchronously add messages to the record in PostgreSQL.
        """
        for message in messages:
            await self.aadd_message(message)

    async def aadd_message(self, message: BaseMessage) -> None:
        """
        Asynchronously append the message to the record in PostgreSQL.
        """
        async_conn = await self._initialize_async_connection()
        async with async_conn.cursor() as cursor:
            await cursor.execute(
                sql.SQL("INSERT INTO {table_name} (session_id, message) VALUES (%s, %s)").format(
                    table_name=sql.Identifier(self._table_name)
                ),
                (self._session_id, _message_to_dict(message))
            )
            await async_conn.commit()


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
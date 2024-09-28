import os
from functools import lru_cache

import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from contextlib import contextmanager, asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from psycopg_pool import AsyncConnectionPool

@lru_cache()
def get_sync_engine():
    return sa.create_engine(os.environ.get("DATABASE_URL"))

@contextmanager
def sync_session_scope():
    with Session(bind=get_sync_engine()) as session:
        try:
            yield session
            session.commit()
        except Exception as exc:
            session.rollback()
            raise exc
        finally:
            session.close()

def get_session():
    with sync_session_scope() as session:
        return session

@lru_cache()
def get_async_engine():
    return create_async_engine(os.environ.get("DATABASE_URL"))

@asynccontextmanager
async def async_session_scope():
    async_session = sessionmaker(
        get_async_engine(), class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            raise exc
        finally:
            await session.close()

async def get_async_session():
    async with async_session_scope() as session:
        return session

@lru_cache()
def create_async_psycopg_pool():
    return AsyncConnectionPool(os.environ.get("DATABASE_URL"))

@asynccontextmanager
async def async_psycopg_connection():
    pool = create_async_psycopg_pool()
    async with pool.connection() as conn:
        try:
            yield conn
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise

async def get_async_psycopg_connection():
    async with async_psycopg_connection() as conn:
        return conn
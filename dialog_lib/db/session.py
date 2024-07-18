import os

import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from contextlib import contextmanager

from functools import cache

@cache
def get_engine():
    return sa.create_engine(os.environ.get("DATABASE_URL"))

@contextmanager
def session_scope():
    with Session(bind=get_engine()) as session:
        try:
            yield session
            session.commit()
        except Exception as exc:
            session.rollback()
            raise exc
        finally:
            session.close()

def get_session():
    with session_scope() as session:
        return session

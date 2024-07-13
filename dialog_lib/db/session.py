import os

import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from contextlib import contextmanager

engine = sa.create_engine(os.environ.get("DATABASE_URL"))
Session = sessionmaker(bind=engine)

@contextmanager
def session_scope():
    with Session(bind=engine) as session:
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
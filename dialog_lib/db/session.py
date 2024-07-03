import os

import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from contextlib import contextmanager

engine = sa.create_engine(os.environ.get("DATABASE_URL"))
Session = sessionmaker(bind=engine)

@contextmanager
def get_session():
    session = Session()
    try:
        yield session
        session.flush()
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
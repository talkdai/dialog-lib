import os

import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from contextlib import contextmanager

engine = sa.create_engine(os.environ.get("DATABASE_URL"))
Session = sessionmaker(bind=engine)

@contextmanager
def get_session():
    with Session() as session:
        try:
            yield session
            session.commit()
        except Exception as exc:
            session.rollback()
            raise exc
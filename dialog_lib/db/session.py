import os

import sqlalchemy as sa
from sqlalchemy.orm import Session

def get_session():
    engine = sa.create_engine(os.environ.get("DATABASE_URL"))
    session = Session(engine)
    yield session
    session.close()
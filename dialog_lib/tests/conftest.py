import os
import pytest
import sqlalchemy

from aioresponses import aioresponses
from sqlalchemy.orm import Session
from dialog_lib.db.models import Base


@pytest.fixture
def db_engine():
    return sqlalchemy.create_engine(os.environ.get('DATABASE_URL'))

@pytest.fixture
def db_session(db_engine):
    Base.metadata.create_all(db_engine)
    session = Session(db_engine)
    yield session
    session.rollback()
    session.close()

@pytest.fixture
def mock_aioresponse():
    with aioresponses() as m:
        yield m
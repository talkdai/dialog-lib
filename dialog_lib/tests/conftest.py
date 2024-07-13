import os
import pytest
import sqlalchemy

from aioresponses import aioresponses
from sqlalchemy.orm import Session
from dialog_lib.db.models import Base
from dialog_lib.db import get_session


@pytest.fixture
def db_engine():
    return sqlalchemy.create_engine(os.environ.get('DATABASE_URL'))


@pytest.fixture
def db_session():
    return get_session()


@pytest.fixture
def mock_aioresponse():
    with aioresponses() as m:
        yield m
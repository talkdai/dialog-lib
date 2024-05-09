import pytest
import responses

from dialog_lib.db.models import CompanyContent
from dialog_lib.loaders.web import load_webpage


def test_load_web_content(mock_aioresponse, db_session, mocker):
    mocker.patch('dialog_lib.loaders.web.generate_embedding', return_value=[0] * 1536)
    mock_aioresponse.get('http://example.com', body='Hello, world!')

    content = load_webpage('http://example.com', db_session, None, 1)
    assert content.question == "Example Domain"
    assert content.embedding == [0] * 1536

    content = db_session.query(CompanyContent).all()
    assert len(content) == 1

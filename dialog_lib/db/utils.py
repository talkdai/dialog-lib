import uuid

from .models import Chat


def create_chat_session(identifier=None, dbsession=None, model=Chat):
    if identifier is None:
        identifier = uuid.uuid4().hex

    chat = dbsession.query(model).filter_by(session_id=identifier).first()
    if not chat:
        chat = model(session_id=identifier)
        dbsession.add(chat)
        dbsession.commit()

    return {"chat_id": chat.session_id}

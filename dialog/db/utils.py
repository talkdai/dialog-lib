import uuid

from dialog.db.models import Chat


def create_session(identifier=None, dbsession=None):
    if identifier is None:
        identifier = uuid.uuid4().hex

    chat = dbsession.query(Chat).filter_by(session_id=identifier).first()
    if not chat:
        chat = Chat(session_id=identifier)
        dbsession.add(chat)
        dbsession.commit()

    return {"chat_id": chat.session_id}

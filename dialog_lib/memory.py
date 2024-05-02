from dialog_lib.db.memory import *
from langchain.memory.buffer import ConversationBufferMemory

def generate_local_memory_instance(memory_key="chat_history"):
    return ConversationBufferMemory(
        memory_key=memory_key,
        return_messages=True
    )


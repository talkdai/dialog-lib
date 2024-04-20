from typing import List

from sqlalchemy import select
from langchain_core.embeddings import Embeddings
from dialog_lib.db.models import CompanyContent


def generate_embeddings(
    documents: List[str], embedding_llm_instance: Embeddings = None
):
    """
    Generate embeddings for a list of documents
    """
    return embedding_llm_instance.embed_documents(documents)


def generate_embedding(document: str, embedding_llm_instance: Embeddings = None):
    """
    Generate embeddings for a single instance of document

    :param document: str - The document/string to generate embeddings for
    :param embedding_llm_instance: - The Embedding LLM instance to use for generating embeddings

    """
    return embedding_llm_instance.embed_query(document)


def get_most_relevant_contents_from_message(
    message,
    top=5,
    dataset=None,
    session=None,
    embeddings_llm=None,
    cosine_similarity_threshold=0.5,
    model=CompanyContent,
    embedding_column="embedding",
):
    message_embedding = generate_embedding(message, embeddings_llm)
    filters = [
        model.embedding.cosine_distance(message_embedding)
        < cosine_similarity_threshold,
    ]

    if dataset is not None:
        filters.append(model.dataset == dataset)

    possible_contents = session.scalars(
        select(model)
        .filter(*filters)
        .order_by(getattr(model, embedding_column).cosine_distance(message_embedding))
        .limit(top)
    ).all()
    return possible_contents

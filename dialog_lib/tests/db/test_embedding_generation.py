from dialog_lib.tests.fixtures.embedding_model import FakeEmbeddingModel
from dialog_lib.embeddings.generate import generate_embedding, generate_embeddings

def test_generate_embedding_for_single_document():
    embedding_model = FakeEmbeddingModel()
    embedding = generate_embedding("Hello, world!", embedding_model)
    assert len(embedding) == 1536

def test_generate_embedding_for_multiple_documents():
    embedding_model = FakeEmbeddingModel()
    embeddings = generate_embeddings(["Hello, world!", "Hello, world 2!"], embedding_model)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1536
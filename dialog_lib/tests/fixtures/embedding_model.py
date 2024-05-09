class FakeEmbeddingModel:
    def embed_documents(self, contents, token_length=1536):
        return_value = []
        for content in contents:
            return_value.append([0] * token_length)

        return return_value

    def embed_query(self, content, token_length=1536):
        return [0] * token_length
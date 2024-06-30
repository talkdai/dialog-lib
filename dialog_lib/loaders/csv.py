from dialog_lib.db.models import CompanyContent
from dialog_lib.embeddings.generate import generate_embedding

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader


def load_csv(
        file_path, dbsession, embeddings_model_instance=None,
        embedding_llm_model=None, embedding_llm_api_key=None, company_id=None
    ):
    loader = CSVLoader(file_path=file_path)
    contents = loader.load()

    if not embeddings_model_instance:
        if embedding_llm_model.lower() == "openai":
            embeddings_model_instance = OpenAIEmbeddings(openai_api_key=embedding_llm_api_key)
        else:
            raise ValueError("Invalid embeddings model")

    for csv_content in contents:
        content = {}

        for line in csv_content.page_content.split("\n"):
            values = line.split(": ")
            content[values[0]] = values[1]

        company_content = CompanyContent(
            category="csv",
            subcategory="csv-content",
            question=content["question"],
            content=content["content"],
            dataset=company_id,
            embedding=generate_embedding(csv_content.page_content, embeddings_model_instance)
        )
        dbsession.add(company_content)

    dbsession.commit()
    return company_content
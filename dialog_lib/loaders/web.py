from dialog_lib.db.models import CompanyContent
from dialog_lib.embeddings.generate import generate_embedding
from dialog_lib.db import get_session
from langchain_community.document_loaders import WebBaseLoader


def load_webpage(url, embeddings_model_instance, session=get_session(), company_id=None):
    loader = WebBaseLoader(url)
    contents = loader.load()

    for url_content in contents:
        company_content = CompanyContent(
            link=url,
            category="web",
            subcategory="website-content",
            question=url_content.metadata["title"],
            content=url_content.page_content,
            dataset=company_id,
            embedding=generate_embedding(url_content.page_content, embeddings_model_instance)
        )
        session.add(company_content)

    return company_content
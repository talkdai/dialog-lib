from typing import TYPE_CHECKING, Any, Dict, List, Optional

from dialog_lib.db.models import CompanyContent
from sqlalchemy.orm import DeclarativeBase, Session

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from dialog_lib.embeddings.generate import get_most_relevant_contents_from_message

class DialogRetriever(BaseRetriever):
    content_model: DeclarativeBase = CompanyContent
    session: Session
    threshold: float = 0.5
    dataset: Optional[Any] = None
    embedding_llm: Optional[Any] = None
    embedding_column: str = "embedding"
    top_k: int = 5

    def _get_relevant_documents(self, query, *, run_manager):
        relevant_contents = get_most_relevant_contents_from_message(
            query,
            top=self.top_k,
            dataset=self.dataset,
            session=self.session,
            embeddings_llm=self.embedding_llm,
            cosine_similarity_threshold=self.threshold,
            model=self.content_model,
            embedding_column=self.embedding_column,
        )
        return [
            Document(
                page_content=f"{content.question}\n\n{content.content}",
                metadata={
                    "title": content.question,
                    "category": content.category,
                    "subcategory": content.subcategory,
                    "dataset": content.dataset,
                    "link": content.link,
                },
            )
            for content in relevant_contents
        ]



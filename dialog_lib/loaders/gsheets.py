import gspread
import logging

from dialog_lib.db.models import CompanyContent
from dialog_lib.db.session import get_session
from dialog_lib.embeddings.generate import generate_embedding

from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union


logger = logging.getLogger(__name__)

class GoogleSheetsLoader(BaseLoader):
    def __init__(self, credentials_path: Union[str, Path], spreadsheet_url: str, sheet_name: str):
        self.sheet_name = sheet_name
        self.gc = gspread.service_account(
            filename=credentials_path
        )
        self.spreadsheet_url = spreadsheet_url
        self.sh = self.gc.open_by_url(self.spreadsheet_url)
        self.worksheet = self.sh.worksheet(self.sheet_name)

    def _read_sheets(self):
        headers = self.worksheet.row_values(1)
        for idx, record in enumerate(self.worksheet.get_all_records()[1:]):
            yield Document(
                page_content="".join([f"{k}: {v}\n" for k, v in record.items()]),
                metadata={
                    "source": "google-sheets",
                    "sheets_url": self.spreadsheet_url,
                    "sheet_name": self.sheet_name,
                }
            )

    def lazy_load(self) -> Iterator[Document]:
        yield from self._read_sheets()


def load_google_sheets(
        credentials_path, spreadsheet_url, sheet_name, dbsession=get_session(),
        embeddings_model_instance=None, embedding_llm_model=None, embedding_llm_api_key=None,
        company_id=None
    ):
    loader = GoogleSheetsLoader(credentials_path, spreadsheet_url, sheet_name)
    contents = loader.load()

    if not embeddings_model_instance:
        if embedding_llm_model.lower() == "openai":
            embeddings_model_instance = OpenAIEmbeddings(openai_api_key=embedding_llm_api_key)
        else:
            raise ValueError("Invalid embeddings model")

    for csv_content in contents:
        content = {}

        for line in csv_content.page_content.split("\n"):
            if line != "":
                values = line.split(": ")
                content[values[0]] = values[1]

        if not dbsession.query(CompanyContent).filter(
            CompanyContent.question == content["question"], CompanyContent.content == content["content"]
        ).first():
            company_content = CompanyContent(
                category="csv",
                subcategory="csv-content",
                question=content["question"],
                content=content["content"],
                dataset=company_id,
                embedding=generate_embedding(csv_content.page_content, embeddings_model_instance)
            )
            dbsession.add(company_content)
        else:
            logger.warning(f"Question: {content['question']} already exists in the database. Skipping.")

    dbsession.commit()
import os
from pathlib import Path
from typing import Dict, Optional, Type

import fire
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chains.question_answering.stuff_prompt import PROMPT
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.excel import UnstructuredExcelLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

load_dotenv()

embeddings = SentenceTransformerEmbeddings(
    model_name=os.getenv("EMBEDDINGS_MODEL"),
)

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=os.getenv("VECTOR_STORE_DIR"),
)

LOADERS: Dict[str, Type[BaseLoader]] = {
    ".txt": TextLoader,
    ".pdf": PyMuPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".csv": CSVLoader,
    ".md": UnstructuredMarkdownLoader,
}


def get_document_loader(file_path: Path) -> Optional[Type[BaseLoader]]:
    return LOADERS.get(file_path.suffix)


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))


def add_doc_to_wiki(
    file_path: Path,
    ignore_unsupported_file: bool = False,
):
    loader_cls = get_document_loader(file_path)
    if loader_cls is None:
        if not ignore_unsupported_file:
            raise ValueError(f"unsupported file type: {file_path}")
        return

    loader = loader_cls(str(file_path))

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    documents = text_splitter.split_documents(documents)

    vector_store.add_documents(documents)


class Operator:
    @property
    def qa(self):
        if not hasattr(self, "_qa"):
            llm = GPT4All(
                model=os.getenv("LLM_MODEL_PATH"),
                verbose=bool(os.getenv("LLM_VERBOSE")),
            )

            self._qa = RetrievalQA.from_chain_type(
                llm,
                retriever=vector_store.as_retriever(),
                chain_type_kwargs={
                    "prompt": PROMPT,
                },
            )
        return self._qa

    def add(
        self,
        path: str,
        ignore_unsupported_file: bool = False,
        recursived: bool = False,
    ):
        file_path = Path(path)
        if not file_path.exists():
            raise ValueError(f"the path {path} does not exist.")
        if file_path.is_dir():
            for sub_file_path in (
                file_path.glob() if recursived else file_path.iterdir()
            ):
                if sub_file_path.is_file():
                    add_doc_to_wiki(sub_file_path, ignore_unsupported_file)
        elif file_path.is_file():
            add_doc_to_wiki(file_path, ignore_unsupported_file)

    def query(
        self,
        query: str,
    ):
        answer = self.qa.run(query)
        print(answer)

    def search(self, query: str, top_k: int = 10):
        docs = vector_store.similarity_search(query=query, k=top_k)

        for doc in docs:
            print(doc.page_content)


if __name__ == "__main__":
    fire.Fire(Operator)

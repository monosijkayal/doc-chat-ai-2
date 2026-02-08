from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import re


def clean_text(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([.,:;!?])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ingest_pdf(pdf_path: str, persist_dir: str = "chroma_db"):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # ✅ CLEAN THE TEXT HERE
    text = clean_text(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=300
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    vectordb.persist()

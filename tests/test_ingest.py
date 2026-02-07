from ingest import ingest_pdf
import os


def test_pdf_ingestion():
    ingest_pdf("data/sample.pdf")
    assert os.path.exists("chroma_db")

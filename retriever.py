from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings



def get_retriever(persist_dir="chroma_db"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    return vectordb.as_retriever(search_kwargs={"k": 3})

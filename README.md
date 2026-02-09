# Internship Interview – RAG Question Answering System (Python)

This project is a **Retrieval-Augmented Generation (RAG)** based **Question Answering system** built in **Python**.  
It answers questions **strictly from a given PDF document** using embeddings, semantic search, and a **local LLM (Ollama)**, and returns a **confidence score** for each answer.

The project supports **two interfaces**:
- 🖥 **CLI (Command Line Interface)**
- 🌐 **FastAPI (REST API)**

This design demonstrates clean separation between **core AI logic** and **application interfaces**, which is ideal for internship interviews.

---

## 📌 Features

- PDF ingestion and text cleaning
- Chunking with overlap for better context
- Vector embeddings using HuggingFace sentence-transformers
- Persistent vector database using ChromaDB
- Semantic similarity search
- Local LLM inference using Ollama
- RAG-based grounded answers (no outside knowledge)
- Confidence score for each answer
- Lightweight conversation memory (last turn only)
- CLI + FastAPI interface using the same core logic

---

## 🛠 Tech Stack (Actual Usage)

- **Python 3.9+**
- **pypdf** – PDF text extraction
- **langchain** – text splitting utilities
- **langchain-community** – Chroma & embeddings
- **chromadb** – vector database
- **sentence-transformers** – embedding model
- **ollama** – local LLM inference
- **fastapi** – REST API layer
- **uvicorn** – ASGI server
- **pytest** (optional) – ingestion test

---

## 📂 Project Structure
.
├── app.py # CLI entry point
├── api.py # FastAPI entry point
├── ingest.py # PDF ingestion & vector store creation
├── retriever.py # Vector store retriever
├── llm.py # Ollama LLM wrapper
├── qa.py # RAG-based question answering logic
├── requirements.txt
├── data/
│ └── sample.pdf # Source document (Inner Game of Tennis)
├── chroma_db/ # Persisted vector database (auto-created)
└── tests/
└── test_ingest.py # Optional ingestion test


---

## ⚙️ Setup Instructions

 1️⃣ Install Ollama (Required)

Install Ollama and pull a model:

```bash
ollama run phi3:mini

(You can change the model name in llm.py if needed.)



2️⃣ Create & Activate Virtual Environment
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows


3️⃣ Install Dependencies
pip install -r requirements.txt


4️⃣ Add Your PDF

Place your document here:

data/sample.pdf

▶️ Running the Project (CLI)
python app.py

Running the Project (FastAPI)

Start the server:

uvicorn api:app --reload


🧪 How to Use the API
Endpoint
POST /ask

Request Body
{
  "question": "What is the Inner Game?"
}

Response
{
  "answer": "The Inner Game refers to ...",
  "confidence": 0.83
}


⚠️ Note:
The root route / is not defined.
Use /docs for interaction.



🔍 How It Works
1. PDF Ingestion (ingest.py)

Extracts text from PDF using pypdf

Cleans malformed spacing

Splits text into overlapping chunks

Generates embeddings using all-MiniLM-L6-v2

Stores vectors in ChromaDB (persistent)

2. Retrieval (retriever.py)

Loads the persisted vector store

Performs semantic similarity search

Returns documents with similarity scores

3. Question Analysis (qa.py)

Detects question type:

Definition questions (what is, what are)

Why questions

General questions

Dynamically adjusts retrieval depth (k)

Rejects answers when similarity is too weak

4. RAG Prompting

Document context is authoritative

Conversation memory is reference-only

LLM is strictly forbidden from using outside knowledge

5. Answer Generation (llm.py)

Sends structured prompt to Ollama

Returns grounded answer text

6. Confidence Scoring

Confidence is calculated from similarity score:

confidence = 1 / (1 + min_similarity_score)


Range:

0.0 → No confidence

1.0 → High confidence


Design Choices 

Local LLM (Ollama)
→ No API keys, offline, reproducible

Persistent Vector Store
→ Ingestion runs once, queries are fast

Minimal Conversation Memory
→ Prevents hallucination

Strict Prompt Hierarchy
→ Document > Memory > Question

Dual Interface (CLI + API)
→ Demonstrates reusable architecture


Future Improvements

Session-based memory (per user)

Multi-PDF ingestion

Source citation in answers

Async LLM calls

Frontend UI (React / Streamlit)


👤 Author

Monosij Kayal
Aspiring Software Engineer
Portfolio: https://monosij.tech
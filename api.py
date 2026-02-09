from fastapi import FastAPI
from pydantic import BaseModel

from ingest import ingest_pdf
from retriever import get_retriever
from llm import OllamaLLM
from qa import answer_question

app = FastAPI(title="Internship Interview RAG API")

# 🔹 Run ingestion once at startup
ingest_pdf("data/sample.pdf")

retriever = get_retriever()
llm = OllamaLLM()

# 🔹 Conversation memory (per session later)
chat_history = []


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    confidence: float


@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    result = answer_question(
        llm=llm,
        retriever=retriever,
        question=payload.question,
        chat_history=chat_history
    )

    chat_history.append((payload.question, result["answer"]))

    return result

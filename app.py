from ingest import ingest_pdf
from retriever import get_retriever
from llm import OllamaLLM
from qa import answer_question

# Run ingestion ONCE (comment out after first successful run)
ingest_pdf("data/sample.pdf")

retriever = get_retriever()
llm = OllamaLLM()

# ✅ Conversation memory
chat_history = []

while True:
    question = input("\nAsk a question (type 'exit' to quit): ")
    if question.lower() == "exit":
        break

    # ✅ Pass memory to QA function
    result = answer_question(llm, retriever, question, chat_history)

    print("\nAnswer:")
    print(result["answer"])

    print("\nConfidence:", result["confidence"])

    # ✅ Save conversation memory (last Q&A)
    chat_history.append((question, result["answer"]))

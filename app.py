from ingest import ingest_pdf
from retriever import get_retriever
from llm import OllamaLLM
from qa import answer_question

# Run ingestion ONCE (comment out after first run if needed)
ingest_pdf("data/sample.pdf")

retriever = get_retriever()
llm = OllamaLLM()

while True:
    question = input("\nAsk a question (type 'exit' to quit): ")
    if question.lower() == "exit":
        break

    answer = answer_question(llm, retriever, question)
    print("\nAnswer:")
    print(answer)

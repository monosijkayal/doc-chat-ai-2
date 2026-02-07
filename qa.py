def answer_question(llm, retriever, question: str):
    docs = retriever.invoke(question)

    if not docs:
        return "Answer not available in the provided document."

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Answer the question ONLY using the context below.
If the answer is not in the context, say:
"Answer not available in the provided document."

Context:
{context}

Question:
{question}
"""

    return llm.generate(prompt)

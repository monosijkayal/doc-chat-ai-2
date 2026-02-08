def answer_question(llm, retriever, question: str):
    """
    Answers a question using RAG.
    Returns:
    - answer
    - confidence score
    """

    q = question.lower()

    # 🔹 Question type detection
    is_why_question = q.startswith("why")
    is_definition_question = q.startswith("what") and (
        "what are" in q or "what is" in q
    )

    # 🔹 Retrieval depth tuning
    if is_why_question:
        k = 8
    elif is_definition_question:
        k = 5
    else:
        k = 6

    results = retriever.similarity_search_with_score(question, k=k)

    if not results:
        return {
            "answer": "Answer not available in the provided document.",
            "confidence": 0.0
        }

    docs = [doc for doc, _ in results]
    scores = [score for _, score in results]
    min_score = min(scores)

    # 🔹 Early-reject ONLY if similarity is VERY weak
    # 🔹 Never early-reject definition or why questions
    if min_score > 1.4 and not (is_why_question or is_definition_question):
        return {
            "answer": "Answer not available in the provided document.",
            "confidence": 0.0
        }

    confidence = round(1 / (1 + min_score), 2)

    context = "\n\n".join(doc.page_content for doc in docs)

    # 🔹 Prompt tuned for definition vs explanation
    prompt = f"""
You are answering questions strictly from the given document.

Rules:
- Use ONLY the information in the context below
- If the question asks for a definition or list, answer directly and concisely
- If the question asks "why", explain using sentences from the document
- Do NOT use outside knowledge
- If the answer is not supported by the document, say:
  "Answer not available in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    answer = llm.generate(prompt)

    return {
        "answer": answer,
        "confidence": confidence
    }

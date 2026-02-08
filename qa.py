def answer_question(llm, retriever, question: str, chat_history: list):
    """
    Answers a question using RAG with lightweight conversation memory.
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

    # 🔹 Authoritative document context (MOST IMPORTANT)
    context = "\n\n".join(doc.page_content for doc in docs)

    # 🔹 VERY LIGHT conversation memory (ONLY last 1 turn)
    history_text = ""
    if chat_history:
        prev_q, prev_a = chat_history[-1]
        history_text = f"""
Previous Question:
{prev_q}

Previous Answer:
{prev_a}
"""

    # 🔹 FINAL PROMPT (CORRECT HIERARCHY)
    prompt = f"""
You are answering questions strictly from the given document.

Document Context (AUTHORITATIVE):
{context}

Conversation Memory (REFERENCE ONLY):
{history_text}

Rules:
- Use ONLY the document context to form answers
- Conversation memory is ONLY to understand references like "they", "this", etc.
- Do NOT use conversation memory as a source of facts
- If the question asks for a definition or list, answer directly and concisely
- If the question asks "why", explain using sentences from the document
- Do NOT use outside knowledge
- If the answer is not supported by the document, say:
  "Answer not available in the provided document."

Current Question:
{question}

Answer:
"""

    answer = llm.generate(prompt)

    return {
        "answer": answer,
        "confidence": confidence
    }

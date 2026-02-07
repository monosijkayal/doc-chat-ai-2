import ollama

response = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": "Reply with OK"}]
)

print(response["message"]["content"])


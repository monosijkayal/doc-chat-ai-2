import ollama


class OllamaLLM:
    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

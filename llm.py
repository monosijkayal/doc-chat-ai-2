import ollama

class OllamaLLM:
    def __init__(self, model="phi3:mini"):
        self.model = model

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
        model=self.model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_ctx": 2048  # reduce context window
        }
)
        return response["message"]["content"]

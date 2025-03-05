"""Test Ollama configuration and basic functionality"""

from langchain_ollama import OllamaLLM
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import time

from src.config import Config


class MeasurementCallback(StreamingStdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.token_count = 0
        self.response_length = 0
        self.start_time = None
        self.complete_response = ""

    def on_llm_start(self, *args, **kwargs):
        self.start_time = time.time()

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.token_count += 1
        self.complete_response += token
        # Still stream to stdout
        print(token, end="", flush=True)

    def on_llm_end(self, *args, **kwargs):
        self.response_length = len(self.complete_response)
        self.time_taken = time.time() - self.start_time

    def get_metrics(self):
        return {
            "tokens": self.token_count,
            "characters": self.response_length,
            "time_seconds": round(self.time_taken, 2),
            "tokens_per_second": round(self.token_count / self.time_taken, 2),
        }


def main():
    OLLAMA_MODEL = Config.ollama_model()
    OLLAMA_URL = Config.ollama_url()

    print("\n-----------------------------")
    print("Testing Ollama configuration:")
    print(f"* Model: {OLLAMA_MODEL}")
    print(f"* URL: {OLLAMA_URL}")

    measurement_cb = MeasurementCallback()

    # Initialize LLM
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        callbacks=[measurement_cb],
    )

    # Test model
    print("\n-----------------------------")
    prompt = "Tel me a good joke"
    print(f"Prompt: {prompt}\n")

    print("\n-----------------------------")
    print("Response:\n")
    llm.invoke(prompt)

    metrics = measurement_cb.get_metrics()
    print("\n-----------------------------")
    print("Metrics:")
    print(f"* Total tokens generated: {metrics['tokens']}")
    print(f"* Total characters: {metrics['characters']}")
    print(f"* Time taken: {metrics['time_seconds']}s")
    print(f"* Tokens per second: {metrics['tokens_per_second']}")


if __name__ == "__main__":
    main()

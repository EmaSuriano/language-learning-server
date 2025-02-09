"""Database connection"""

import os

from rag.rag_language_retrieval import RAGLanguageEvaluator

# TODO: move it to env
OLLAMA_URL = "http://localhost:11434"

persist_directory = os.path.join(os.path.dirname(__file__), "language_examples_db")

# Initialize RAG evaluator
rag_evaluator = RAGLanguageEvaluator(
    persist_directory=persist_directory, base_url=OLLAMA_URL
)


def get_rag_evaluator():
    return rag_evaluator

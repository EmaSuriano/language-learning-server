"""Vector store initialization for language examples"""

import json
import os
from pathlib import Path
import shutil
from typing import List

from rag.rag_language_retrieval import (
    RAGLanguageEvaluator,
    LanguageExample,
)

OLLAMA_URL = "http://localhost:11434"


def load_examples_from_directory(
    examples_dir: str = "data/language_examples",
) -> List[LanguageExample]:
    """Load all JSON files from the examples directory"""
    examples = []
    examples_path = Path(examples_dir)

    if not examples_path.exists():
        raise FileNotFoundError(f"Examples directory not found: {examples_dir}")

    # Load each JSON file in the directory
    for json_file in examples_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Convert each example to LanguageExample
                file_examples = [LanguageExample(**example) for example in data]
                examples.extend(file_examples)
                print(f"Loaded {len(file_examples)} examples from {json_file.name}")
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")

    return examples


def reset_vector_store():
    """Reset and initialize the vector store with language examples"""
    persist_directory = "./language_examples_db"

    # Remove existing vector store if it exists
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Removed existing vector store at {persist_directory}")

    # Initialize RAG evaluator
    rag_evaluator = RAGLanguageEvaluator(
        persist_directory=persist_directory, base_url=OLLAMA_URL
    )

    # Load examples from JSON files
    examples = load_examples_from_directory()

    if not examples:
        print("No examples found! Please check the data directory.")
        return

    # Add examples to vector store
    rag_evaluator.add_examples(examples)
    print(f"Added {len(examples)} examples to the vector store")


if __name__ == "__main__":
    print("Initializing vector store...")
    reset_vector_store()
    print("Done!")

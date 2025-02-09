from typing import List, Optional
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel  # Add this import

OLLAMA_URL = "http://localhost:11434"


class LanguageExample(BaseModel):
    """Example of language usage for a specific CEFR level"""

    phrase: str
    level: str  # CEFR level (A1-C2)
    context: str  # situation where this phrase would be appropriate


class RAGLanguageEvaluator:
    def __init__(
        self,
        persist_directory,
        base_url: str = OLLAMA_URL,
    ):
        self.embedding_model = OllamaEmbeddings(
            model="nomic-embed-text",  # You can also use other models like "llama2"
            base_url=base_url,
        )
        self.persist_directory = persist_directory
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
        )

    def add_examples(self, examples: List[LanguageExample]):
        documents = []
        metadatas = []

        for example in examples:
            doc_text = f"{example.phrase}\n\nContext: {example.context}"
            documents.append(doc_text)

            metadata = {
                "level": example.level,
                "phrase": example.phrase,
            }
            metadatas.append(metadata)

        self.vectorstore.add_texts(texts=documents, metadatas=metadatas)

    def get_relevant_examples(
        self, query: str, level: str, category: Optional[str] = None, k: int = 5
    ) -> List[LanguageExample]:
        filter_dict = {"level": level}
        if category:
            filter_dict["category"] = category

        results = self.vectorstore.similarity_search_with_score(
            query=query, k=k, filter=filter_dict
        )

        examples = []
        for doc, score in results:
            metadata = doc.metadata
            example = LanguageExample(
                phrase=metadata["phrase"],
                level=metadata["level"],
                context=doc.page_content.split("\n\nContext: ")[1],
            )
            examples.append(example)

        return examples

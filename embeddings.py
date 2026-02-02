from typing import List
import numpy as np
import ollama


class EmbeddingModel:
    """wrapper for embedding generation"""

    def __init__(self, model: str = 'nomic-embed-text'):
        self.model = model

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            return response['embedding']
        except Exception as e:
            # fallback to random embedding if ollama fails
            print(f"error generating embedding: {e}!!")
            return list(np.random.randn(384))

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.get_embedding(text) for text in texts]

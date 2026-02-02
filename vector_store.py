import chromadb
from typing import List, Dict
import uuid


class PaperVectorStore:
    """chromadb store for paper embeddings"""
    def __init__(self, persist_directory: str = './chroma_db'):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name='papers',
            # using cosine metric
            metadata={'hnsw:space': 'cosine', 'description': 'scientific papers embeddings'},
        )

    def add_papers(self, papers: List[Dict], embeddings: List[List[float]]) -> None:
        """add papers with embeddings to vector store"""
        if not papers or not embeddings:
            return

        # random ids
        ids = [str(uuid.uuid4()) for _ in papers]
        documents = []
        metadatas = []

        for i, paper in enumerate(papers):
            # combine title and summary for search
            doc_text = f"{paper['title']} {paper['summary'][:1000]}"
            documents.append(doc_text)

            metadata = {
                'title': paper['title'],
                'source': paper['source'],
                'published': paper['published'],
                'categories': ','.join(paper['categories'][:3]),
                'query_relevance': str(paper.get('relevance_score', 0))
            }
            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def find_similar(self, query_embedding: List[float], n_results: int = 5) -> List[Dict]:
        """find similar papers using vector similarity"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        similar_papers = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                similar_papers.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })

        return similar_papers

    def get_all_papers_count(self) -> int:
        return self.collection.count()
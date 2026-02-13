import chromadb
from typing import List, Dict, Any
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

    @staticmethod
    def filter_list_by_ids(orig_list: List[Any], ids_to_keep: List[int]) -> List[Any]:
        filtered_list = [orig_list[i] for i in range(len(orig_list)) if i in ids_to_keep]
        return filtered_list

    def add_papers(self, papers: List[Dict], embeddings: List[List[float]]) -> None:
        """add papers with embeddings to vector store"""
        if not papers or not embeddings:
            return
        assert len(papers) == len(embeddings)

        # random ids
        ids = [str(uuid.uuid4()) for _ in papers]
        documents = []
        metadatas = []
        ids_to_keep = []

        for i, paper in enumerate(papers):
            # combine title and summary for search
            doc_text = f"{paper['title']} {paper['summary'][:1000]}"
            metadata = {
                'title': paper['title'],
                'source': paper['source'],
                'published': paper['published'],
                'categories': ','.join(paper['categories'][:3]),
                'query_relevance': str(paper.get('relevance_score', 0))
            }
            documents.append(doc_text)
            metadatas.append(metadata)

            # find most similar paper, very high similarity -> duplicate, skip adding
            sim_papers = self.find_similar(query_embedding=embeddings[i], n_results=1)
            if sim_papers and sim_papers[0]['distance'] <= 0.001:
                continue
            else:
                ids_to_keep.append(i)

        ids = PaperVectorStore.filter_list_by_ids(ids, ids_to_keep)
        embeddings = PaperVectorStore.filter_list_by_ids(embeddings, ids_to_keep)
        documents = PaperVectorStore.filter_list_by_ids(documents, ids_to_keep)
        metadatas = PaperVectorStore.filter_list_by_ids(metadatas, ids_to_keep)

        if ids:
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
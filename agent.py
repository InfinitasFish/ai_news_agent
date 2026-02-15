from typing import List, Dict
import torch
import numpy as np

from analyzer import PaperAnalyzer
from post_gen import PostGenerator
from parsers import ArxivSource
from vector_store import PaperVectorStore
from embeddings import EmbeddingModel

class ResearchAgent:

    def __init__(self,model: str = 'qwen3:4b', seed: int = 59, sources: List[str] = None,
                 use_vector_store: bool = False):

        self.model = model
        self.seed = seed
        self.sources = sources
        self.use_vector_store = use_vector_store

        self.analyzer = PaperAnalyzer(model=model, seed=seed, use_embeddings=use_vector_store)
        self.generator = PostGenerator()

        if use_vector_store:
            self.vector_store = PaperVectorStore()

        torch.manual_seed(seed)
        np.random.seed(seed)

    # TODO: add search with empty query
    def run_daily_research(self, query: str = 'llm+interpretability', max_papers_per_source: int = 30,
                           top_k: int = 5, hours_back: int = 24, use_semantic_search: bool = True) -> Dict:
        """ Args:
            query: research query
            max_papers_per_source: papers to fetch per source
            top_k: number of top papers to select
            hours_back: hours to look back
            use_semantic_search: use chroma_db and semantic search
        """
        print(f"Research for query: '{query}'")
        print(f"Looking for papers from last {hours_back} hours")

        all_papers = []
        for source in self.sources:
            if source.lower() == 'arxiv':
                print("Retrieving papers from arXiv...")
                papers = ArxivSource.fetch_recent_papers(
                    query=query,
                    max_results=max_papers_per_source,
                    hours_back=hours_back
                )
                print(f"\tFound {len(papers)} recent papers from arXiv")
                all_papers.extend(papers)

        if not all_papers:
            return {"error": "No papers found", "post": ""}

        # analyze with llm
        print("Analyzing papers with LLM...")
        selected_papers = self.analyzer.analyze_with_llm(all_papers, query, top_k=top_k)

        # find sim papers in chroma_db
        print("Finding similar papers in ChromaDB")
        similar_papers = self.find_similar_papers(query, n_results=4)

        # save found papers to chroma_db
        if self.use_vector_store and selected_papers:
            embeddings = []
            for paper in selected_papers:
                if 'embedding' in paper:
                    embeddings.append(paper['embedding'])

            if embeddings:
                self.vector_store.add_papers(selected_papers, embeddings)
                stored_count = self.vector_store.get_all_papers_count()
                print(f'stored papers in vector database (total: {stored_count})')

        print("Generating daily post...")
        post = self.generator.generate_daily_post(selected_papers, similar_papers, query)

        result = {
            "total_papers_found": len(all_papers),
            "selected_papers": len(selected_papers),
            "query": query,
            "papers": selected_papers,
            "post": post
        }

        return result

    def find_similar_papers(self, query: str, n_results: int = 5) -> List[Dict]:
        """rag search, find similar papers using vector search"""
        if not self.use_vector_store:
            return []

        embedder = EmbeddingModel()
        query_embedding = embedder.get_embedding(query)

        similar = self.vector_store.find_similar(query_embedding, n_results)
        return similar

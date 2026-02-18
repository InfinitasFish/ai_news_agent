from typing import List, Dict
from datetime import datetime
import ollama
import numpy as np
from pydantic import BaseModel

from embeddings import EmbeddingModel


class PaperAnalyzer:
    """analyze and rank papers based on relevance"""

    def __init__(self, model: str = 'qwen3:4b', seed: int = 59, use_embeddings: bool = True):
        self.model = model
        self.seed = seed
        self.use_embeddings = use_embeddings
        if use_embeddings:
            self.embedder = EmbeddingModel()

    @staticmethod
    def calculate_relevance(paper: Dict, query: str) -> float:
        """calculate relevance using query-to-text word matching"""

        # keyword matching
        text_to_analyze = f"{paper['title']} {paper['summary']}".lower()
        query_terms = query.lower().split('+')
        matches = sum(1 for term in query_terms if term in text_to_analyze)

        base_score = min(matches / len(query_terms), 1.0) if query_terms else 0.5

        hours_ago = (datetime.utcnow() - paper['published_dt']).total_seconds() / 3600
        # magic numbers
        recency_boost = 1.0 if hours_ago <= 12 else 0.9

        return base_score * recency_boost

    def calculate_semantic_relevance(self, query_embedding: List[float], paper_embedding: List[float]) -> float:
        """calculate relevance using vector similarity"""

        if not self.use_embeddings: return 0.0

        # cosine sim
        query_norm = np.linalg.norm(np.array(query_embedding))
        paper_norm = np.linalg.norm(np.array(paper_embedding))

        if query_norm == 0 or paper_norm == 0:
            return 0.0

        # normalize to 0-1 range
        similarity = np.dot(query_embedding, paper_embedding) / (query_norm * paper_norm)
        return (similarity + 1) / 2


    def analyze_with_llm(self, papers: List[Dict], query: str, top_k: int = 10,
                         use_semantic: bool = True) -> List[Dict]:
        """ analyze papers with llm
            papers: List of paper dictionaries
            query: User query for relevance
            top_k: Number of top papers to return
            use_semantic: Use chroma_db semantic search
        """
        if not papers:
            return []

        if use_semantic and self.use_embeddings:
            # generate embeddings for semantic relevance
            query_embedding = self.embedder.get_embedding(query)
            for paper in papers:
                paper_text = f"{paper['title']} {paper['summary']}"
                paper_embedding = self.embedder.get_embedding(paper_text)
                semantic_score = self.calculate_semantic_relevance(query_embedding, paper_embedding)
                # combine with keyword score
                keyword_score = self.calculate_relevance(paper, query)
                # magic weights for scores
                paper['relevance_score'] = 0.8 * semantic_score + 0.2 * keyword_score
                paper['embedding'] = paper_embedding
        else:
            for paper in papers:
                paper['relevance_score'] = self.calculate_relevance(paper, query)

        sorted_papers = sorted(papers, key=lambda x: x['relevance_score'], reverse=True)
        top_papers = sorted_papers[:min(top_k, len(sorted_papers))]

        analyzed_papers = []
        for paper in top_papers:
            if not paper.get('full_text'):
                continue

            full_text = paper['full_text'][:13000]
            prompt = f"""You're a research assistant. Analyze this paper and write concise summaries for each major section you can identify.

                Title: {paper['title']}
                Categories: {', '.join(paper['categories'])}
                Authors: {', '.join(paper['authors'][:3])}{' et al.' if len(paper['authors']) > 3 else ''}
    
                Read through the paper and provide a brief summary of each major section (like Abstract, Introduction, Methods, Results, Discussion, Conclusion - or whatever structure you find).
    
                Full text:
                {full_text}
            """

            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'seed': self.seed, 'temperature': 0.3}
                )

                paper['analysis'] = response['message']['content'].strip()

            except Exception as e:
                print(f'Error analyzing paper {paper["title"][:50]}...: {e}')
                paper['analysis'] = "Analysis failed due to something"
                paper['analysis_error'] = str(e)

            analyzed_papers.append(paper)

        return analyzed_papers


# deprecated (
class SelectedPapersFormat(BaseModel):
    selected_indices: List[int]
    reasons: List[str]

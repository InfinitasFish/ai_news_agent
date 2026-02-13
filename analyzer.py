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
                paper_text = f"{paper['title']} {paper['summary'][:1000]}"
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
        top_papers = sorted_papers[:min(top_k * 2, len(sorted_papers))]

        papers_context = "\n\n".join([
            f"Paper {i + 1}:\ntitle: {p['title']}\nSummary: {p['summary'][:500]}...\n"
            f"Categories: {', '.join(p['categories'])}\n "

            # not using relevance score in prompt because model just fallback to it mindlessly (at least small 4b model)
            #f"relevance score: {p['relevance_score']:.2f}"

            for i, p in enumerate(top_papers)
        ])

        prompt = f"""You're a helpful research assistant, analyze these scientific papers and select the top {top_k} 
        most relevant to the query: "{query}"

        Papers:
        {papers_context}

        Instructions:
            Consider both relevance to query and significance of findings;
            Prioritize novel or groundbreaking research;
            Consider paper quality and source reputation;
            Return JSON with list of selected paper indices (1-based) and brief reasons;

        Answer in specified json format, e.g. {{"selected_indices": [1, 3, 5], "reasons": ["reason1", "reason2", ...]}}
        """

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'seed': self.seed, 'temperature': 0.3},
                format=SelectedPapersFormat.model_json_schema()
            )
        except Exception as e:
            if hasattr(e, 'message'): msg = e.message
            else: msg = e
            print(f'Error from .chat(), model: {self.model}\n {msg} ')

            # fallback to sort papers by simple relevance
            return top_papers[:top_k]

        # extract json answer
        selected_indices = SelectedPapersFormat.model_validate_json(response['message']['content']).selected_indices
        reasons = SelectedPapersFormat.model_validate_json(response['message']['content']).reasons
        # successful extraction
        if selected_indices and reasons and len(selected_indices) == len(reasons):
            # get 0-based indices and select papers
            selected_papers = []
            # костыль чтобы сметчить индексы пейперов и лист причин
            for reason_idx, paper_idx in enumerate(selected_indices[:top_k]):
                if 0 < paper_idx <= len(top_papers):
                    paper = top_papers[paper_idx - 1].copy()
                    reason = reasons[reason_idx]
                    paper['selection_reason'] = reason
                    selected_papers.append(paper)

            return selected_papers

        # fallback to sort papers by simple relevance
        return top_papers[:top_k]


class SelectedPapersFormat(BaseModel):
    selected_indices: List[int]
    reasons: List[str]

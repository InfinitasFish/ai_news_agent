from typing import List, Dict
from datetime import datetime

SECTION_DELIMITER = '#===================================================#\n\n'

class PostGenerator:
    """generates post from selected papers"""

    @staticmethod
    def generate_daily_post(papers: List[Dict], similar_db_papers: List[Dict], query: str = "llm") -> str:
        if not papers:
            return "No relevant papers found for today."

        post = f"""### Daily Research: {datetime.now().strftime('%Y-%m-%d')}
### Query: "{query}"
### Found *{len(papers)}* relevant papers.
{SECTION_DELIMITER}           
"""

        for i, paper in enumerate(papers, 1):
            post += f"""
Paper: {paper['title']}

Authors: {', '.join(paper['authors'][:3])}{' et al.' if len(paper['authors']) > 3 else ''}

Categories: {', '.join(paper['categories'][:3])}

Published: {paper['published_dt'].strftime('%Y-%m-%d %H:%M UTC')}

Source: {paper['source'].title()}
            
Relevance Score: {paper.get('relevance_score', -1.0):.3f}/1.00
            
Llm analysis: {paper.get('analysis', '')}
            
[Read full paper]({paper['id']})

{SECTION_DELIMITER}
"""

        # chromaDB sim papers
        if similar_db_papers:
            post += f"\nPapers similar in chroma_db to '{query}':\n\n"
            for i, sim_paper in enumerate(similar_db_papers, 1):
                post += f"{i}) {sim_paper['metadata']['title']}.\nSimilarity: {1 - sim_paper['distance']:.4f}\n\n"
        else:
            post += "\nno similar papers found (vector store might be empty)\n\n"

        post += f"{SECTION_DELIMITER}\n\n"

        # additional info
        post += "Additional takeaways:\n"
        topics = set()
        for paper in papers:
            topics.update(paper['categories'][:2])
        post += f"Trending categories: {', '.join(list(topics)[:10])}\n"

        sources = {}
        for paper in papers:
            sources[paper['source']] = sources.get(paper['source'], 0) + 1

        post += "Sources count: " + ", ".join([f"{k}: {v} papers" for k, v in sources.items()]) + "\n"

        # Most active field
        if papers and any(p.get('categories') for p in papers):
            all_cats = [p['categories'][0] for p in papers if p.get('categories')]
            if all_cats:
                most_active = max(set(all_cats), key=all_cats.count)
                post += f"\nMost active fields: {most_active}\n"

        post += f"\nThis post is AI-generated. {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*"

        return post

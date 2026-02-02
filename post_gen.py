from typing import List, Dict
from datetime import datetime
import textwrap


class PostGenerator:
    """generates post from selected papers"""

    @staticmethod
    def generate_daily_post(papers: List[Dict], similar_db_papers: List[Dict], query: str = "general research") -> str:
        if not papers:
            return "No relevant papers found for today."

        post = textwrap.dedent(f"""
            ### Daily Research: {datetime.now().strftime('%Y-%m-%d')}
            ### Query: "{query}"
            ### Found *{len(papers)}* relevant papers.
            #===================================================#\n\n
        """)

        for i, paper in enumerate(papers, 1):
            post += textwrap.dedent(f"""
                Paper: {paper['title']}\n
                Authors: {', '.join(paper['authors'][:3])}{' et al.' if len(paper['authors']) > 3 else ''}\n
                Categories: {', '.join(paper['categories'][:3])}\n
                Published: {paper['published_dt'].strftime('%Y-%m-%d %H:%M UTC')}\n
                Source: {paper['source'].title()}\n
                Summary: {paper['summary'][:400]}... \n
                #===================================================#

                Why it matters: {paper.get('selection_reason', 'error retrieving reason')}
                
                Relevance Score: {paper.get('relevance_score', -1.0):.2f}/1.00

                [Read more]({paper['id']})\n
                #===================================================#\n
            """)

        # chromaDB sim papers
        if similar_db_papers:
            post += f"Papers similar in chroma_db to '{query}':\n\n"
            for i, sim_paper in enumerate(similar_db_papers, 1):
                post += f"{i}) {sim_paper['metadata']['title']}.\nSimilarity: {1 - sim_paper['distance']:.4f}\n\n"
        else:
            post += "no similar papers found (vector store might be empty)\n\n"
        post += textwrap.dedent("#===================================================#\n\n")

        # additional info
        post += """Additional takeaways:\n"""
        topics = set()
        for paper in papers:
            topics.update(paper['categories'][:2])
        post += f"Trending categories: {', '.join(list(topics)[:10])}\n"

        sources = {}
        for paper in papers:
            sources[paper['source']] = sources.get(paper['source'], 0) + 1

        post += f"Sources count:" + ", ".join([f"{k}: {v} papers" for k, v in sources.items()]) + "\n"

        post += textwrap.dedent(f"""
                    Most active fields: {max(set([p['categories'][0] for p in papers if p['categories']]), 
                                           key=[p['categories'][0] for p in papers if p['categories']].count)}"

                    #===================================================#          
                    
                    This post is AI-generated. {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*
                """)

        return post

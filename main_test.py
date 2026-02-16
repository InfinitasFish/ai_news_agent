from datetime import datetime
from agent import ResearchAgent


def main_search(query):
    agent = ResearchAgent(
        model='qwen3:4b',
        seed=59,
        sources=['arxiv'],
        use_vector_store=True,
    )

    if query is None:
        queries_examples = [
            "deep learning neural networks",
            "quantum computing",
            "computer vision",
        ]
        for i, q in enumerate(queries_examples):
            queries_examples[i] = '+'.join(q.split())
        test_query = queries_examples[0]
    else:
        test_query = '+'.join(query.split())

    results = agent.run_daily_research(
        query=test_query,
        max_papers_per_source=30,
        top_k=5,
        hours_back=64,
        use_semantic_search=True,
    )

    print("\n" + "*" * 80)
    print("RESULTS SUMMARY")
    print("*" * 80)
    print(f"Total papers found: {results['total_papers_found']}")
    print(f"Selected papers: {results['selected_papers']}")
    print(f"Query: {results['query']}")

    print("\n" + "*" * 80)
    print("GENERATED POST")
    #print(results['post'])
    print("*" * 80)

    # save results
    with open(f"research_digest_{datetime.now().strftime('%Y%m%d')}.md", "w", encoding="utf-8") as f:
        f.write(results['post'])

    print(f"\nResults saved to research_digest_{datetime.now().strftime('%Y%m%d')}.md")


if __name__ == "__main__":
    main_search('deep computer vision')

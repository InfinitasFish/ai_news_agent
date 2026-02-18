from datetime import datetime, timedelta
from typing import List, Dict
import xml.etree.ElementTree as ET
import urllib.request as libreq
import io
import PyPDF2
import ssl


class ArxivSource:
    """fetcher arxiv papers"""

    ARXIV_API_URL = "http://export.arxiv.org/api/query"
    NAMESPACES = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}

    @staticmethod
    def fetch_recent_papers(query: str = "all",
                            max_results: int = 50,
                            hours_back: int = 24,
                            fetch_full_text: bool = True) -> List[Dict]:

        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        query_url = f"{ArxivSource.ARXIV_API_URL}?search_query={query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"

        papers = []
        response = None
        root = None
        gcontext = ssl.SSLContext()
        try:
            with libreq.urlopen(query_url, context=gcontext) as url:
                response = url.read()
                root = ET.fromstring(response)
        except Exception as e:
            raise RuntimeError(f'Error fetching from arXiv: {e}')

        for entry in root.findall('atom:entry', ArxivSource.NAMESPACES):
            title = entry.find('atom:title', ArxivSource.NAMESPACES).text
            title = ' '.join(title.split()) if title else ""

            summary = entry.find('atom:summary', ArxivSource.NAMESPACES).text
            summary = ' '.join(summary.split()) if summary else ""

            published = entry.find('atom:published', ArxivSource.NAMESPACES).text
            published_dt = datetime.fromisoformat(published.replace('Z', '+00:00'))
            published_dt = published_dt.replace(tzinfo=None)

            # skip papers older than cutoff
            if published_dt < cutoff_time:
                continue

            categories = [cat.attrib['term'] for cat in
                          entry.findall('atom:category', ArxivSource.NAMESPACES)]
            authors = [aut.find('atom:name', ArxivSource.NAMESPACES).text
                       for aut in entry.findall('atom:author', ArxivSource.NAMESPACES)]
            paper_id = entry.find('atom:id', ArxivSource.NAMESPACES).text
            arxiv_id = paper_id.split('/')[-1]
            paper_data = {
                'id': paper_id,
                'title': title,
                'summary': summary,
                'categories': categories,
                'authors': authors,
                'published': published,
                'published_dt': published_dt,
                'source': 'arxiv'
            }

            if fetch_full_text:
                try:
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    with libreq.urlopen(pdf_url, context=gcontext) as pdf_response:
                        pdf_content = pdf_response.read()

                        pdf_file = io.BytesIO(pdf_content)
                        pdf_reader = PyPDF2.PdfReader(pdf_file)

                        full_text = []
                        for page in pdf_reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                full_text.append(page_text)

                        full_text_join = ' '.join(full_text)
                        full_text_join = ' '.join(full_text_join.split())
                        paper_data['full_text'] = full_text_join

                except Exception as e:
                    print(f'Error: Could not fetch full text for {arxiv_id}: {e}')
                    paper_data['full_text'] = ''

            papers.append(paper_data)

        return papers


if __name__ == '__main__':
    for paper in ArxivSource.fetch_recent_papers('llm+rag', hours_back=32):
        print(paper)


# telegram bot deployment

import logging
import os
from typing import List
import textwrap

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from dotenv import load_dotenv

from agent import ResearchAgent
from post_gen import SECTION_DELIMITER


CHUNK_SIZE = 1900


def split_post_by_chunks(full_post: str, chunk_size: int) -> List[str]:
    chunks = []
    chunk_count = len(full_post) // chunk_size
    for i in range(chunk_count):
        chunks.append(full_post[i * chunk_size:(i + 1) * chunk_size])
    return chunks


def split_post_by_delimiter(full_post, delimiter=SECTION_DELIMITER):
    return [chunk.strip() for chunk in full_post.split(SECTION_DELIMITER)]


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start_message = textwrap.dedent("""
        This bot parses arXiv papers that were published for the last N hours given the query.
        
        Usage: /search_arxiv query n_hours
        """)

    await context.bot.send_message(chat_id=update.effective_chat.id, text=start_message)


# bot accepts n_hours and query for search
async def search_arxiv_last_n_by_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # args are passed automatically, then split by using whitespace
    # print(context.args)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f'Searching in progress...')
    try:
        n_hours = int(context.args[-1])
        query = '+'.join(context.args[:-1])
    except Exception as e:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f'Invalid arguments in /search_arxiv : {e}')
        return

    agent = ResearchAgent(
        model='qwen3:4b',
        seed=59,
        sources=['arxiv'],
        use_vector_store=True,
    )

    results = agent.run_daily_research(
        query=query,
        hours_back=n_hours,
        max_papers_per_source=10,
        top_k=5,
        use_semantic_search=True,
    )

    if results['total_papers_found'] == 0:
        await context.bot.send_message(chat_id=update.effective_chat.id, text='No papers were found, try different query or bigger n_hours')
        return
    post = results['post']
    post_sections = split_post_by_delimiter(post)
    post_sections_chunks = []
    for section in post_sections:
        post_sections_chunks.extend(split_post_by_chunks(section))
    print(len(post), len(post_sections), len(post_sections_chunks))
    for chunk in post_sections_chunks:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=chunk)


if __name__ == '__main__':

    load_dotenv()
    bot_token = os.environ['TG_BOT_TOKEN']
    application = ApplicationBuilder().token(bot_token).build()

    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)
    search_arxiv_handler = CommandHandler('search_arxiv', search_arxiv_last_n_by_query, has_args=True)
    application.add_handler(search_arxiv_handler)

    application.run_polling()


# telegram bot deployment

import logging
import os

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from dotenv import load_dotenv

from agent import ResearchAgent


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")


# bot accepts n_hours and query for search
async def search_arxiv_last_n_by_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # args are passed automatically, then split by using whitespace
    # print(context.args)

    n_hours = context.args[-1]
    query = '+'.join(context.args[:-1])

    agent = ResearchAgent(
        model='qwen3:4b',
        seed=59,
        sources=['arxiv'],
        use_vector_store=True,
    )

    results = agent.run_daily_research(
        query=query,
        hours_back=int(n_hours),
        max_papers_per_source=30,
        top_k=5,
        use_semantic_search=True,
    )

    await context.bot.send_message(chat_id=update.effective_chat.id, text=results['post'])


if __name__ == '__main__':

    load_dotenv()
    bot_token = os.environ['TG_BOT_TOKEN']
    application = ApplicationBuilder().token(bot_token).build()

    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)
    search_arxiv_handler = CommandHandler('search_arxiv', search_arxiv_last_n_by_query, has_args=True)
    application.add_handler(search_arxiv_handler)

    application.run_polling()


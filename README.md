Телеграм бот, который умеет парсить актуальные статьи с arXiv за последние N часов по заданной query. Для оценки релеватности используется word-matching и синусное сходство query и текста статьи. В суммаризации также участвует llm. 
Найденные статьи также сохраняются в векторную базу данных, чтобы переиспользовать их в новых запросах.


Основные инструменты: ollama, chromaDb, python-telegram-bot


<img width="598" height="363" alt="image" src="https://github.com/user-attachments/assets/0014b707-5468-4440-96e4-4e0ca123e450" />


<img width="461" height="780" alt="image" src="https://github.com/user-attachments/assets/d48e24f1-e597-41eb-81ab-615960b48e6f" />


<img width="451" height="296" alt="image" src="https://github.com/user-attachments/assets/1f81a91d-a0d8-4bcf-b7a0-d5bdad728e8b" />

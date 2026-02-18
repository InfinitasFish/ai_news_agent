Телеграм бот, который умеет парсить актуальные статьи с arXiv за последние N часов по заданной query. Для оценки релеватности используется word-matching и синусное сходство query и текста статьи. В суммаризации также участвует llm. 
Найденные статьи также сохраняются в векторную базу данных, чтобы переиспользовать их в новых запросах.


Основные инструменты: ollama, chromaDb, python-telegram-bot


<img width="598" height="363" alt="image" src="https://github.com/user-attachments/assets/0014b707-5468-4440-96e4-4e0ca123e450" />


<img width="442" height="573" alt="image" src="https://github.com/user-attachments/assets/d7b6be49-6ea9-47d9-ae9b-8f8a7a9a6c82" />


<img width="451" height="296" alt="image" src="https://github.com/user-attachments/assets/1f81a91d-a0d8-4bcf-b7a0-d5bdad728e8b" />

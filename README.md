Телеграм бот, который умеет парсить актуальные статьи с arXiv за последние N часов по заданной query. Для оценки релеватности используется word-matching и синусное сходство query и текста статьи. В суммаризации также участвует llm. 
Найденные статьи также сохраняются в векторную базу данных, чтобы переиспользовать их в новых запросах.


Основные инструменты: ollama, chromaDb, python-telegram-bot


<img width="454" height="773" alt="image" src="https://github.com/user-attachments/assets/0c49bef0-7ff8-4223-9aa1-ecbdadf7c29c" />


<img width="446" height="487" alt="image" src="https://github.com/user-attachments/assets/05c9ff4d-27fc-4624-bef9-831f5cad54b8" />



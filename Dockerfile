# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt requirements.txt

# Устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --default-timeout=300

# Копируем все файлы приложения в контейнер
COPY ./app /app/app 
COPY main.py /app/main.py
COPY .env /app/.env

# Указываем FastAPI приложению порт, который будет слушать uvicorn
EXPOSE 8000

# Переменные окружения для кешей моделей
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_CACHE=/app/.cache/huggingface/hub
ENV TORCH_HOME=/app/.cache/torch

# Создаем директории для кеша и даем права на запись
RUN mkdir -p /app/.cache/sentence_transformers /app/.cache/huggingface/hub /app/.cache/torch && \
    chmod -R 777 /app/.cache

# Команда для запуска приложения
# Uvicorn ожидает путь в формате module:variable
# Если main.py лежит в /app, то модуль будет 'main' (т.к. /app в PYTHONPATH)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --default-timeout=300

COPY ./app /app/app 
COPY main.py /app/main.py
COPY .env /app/.env

EXPOSE 8000

ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_CACHE=/app/.cache/huggingface/hub
ENV TORCH_HOME=/app/.cache/torch

RUN mkdir -p /app/.cache/sentence_transformers /app/.cache/huggingface/hub /app/.cache/torch && \
    chmod -R 777 /app/.cache

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
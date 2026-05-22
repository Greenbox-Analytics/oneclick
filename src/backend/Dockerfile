FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==2.3.3

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
 && poetry install --no-root --without dev --no-interaction --no-ansi

COPY . .

EXPOSE 8080
ENV PORT=8080

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}

FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY apps/api /app/apps/api
COPY artifacts /app/artifacts

RUN pip install --no-cache-dir -e /app

WORKDIR /app/apps/api

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

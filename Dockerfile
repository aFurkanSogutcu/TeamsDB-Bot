# ====== Builder ======
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip wheel && pip wheel --no-cache-dir --no-deps -r requirements.txt -w wheels

# ====== Runtime ======
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN useradd -m appuser
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir /wheels/*

COPY . /app

EXPOSE 8000
USER appuser

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]

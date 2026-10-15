FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from utils.rag_engine import warm_up; warm_up()"

ENV HF_HUB_OFFLINE=1

EXPOSE 8000
# One worker: analysis jobs are held in this process's memory.
CMD ["gunicorn", "-w", "1", "--threads", "8", "--timeout", "120", "-b", "0.0.0.0:8000", "app:app"]

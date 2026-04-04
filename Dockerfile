# Hugging Face Spaces & local Docker
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Gradio /web is optional; false keeps HF Docker builds smaller and startup faster.
ENV ENABLE_WEB_INTERFACE=true

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY delivery_env ./delivery_env
COPY server ./server
COPY openenv.yaml ./openenv.yaml

EXPOSE 7860

# HF Spaces injects PORT; default 7860
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]

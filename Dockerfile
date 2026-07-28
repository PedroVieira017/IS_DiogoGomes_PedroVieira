FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/VisuLogic/VisuLogic-Eval

WORKDIR /app

COPY requirements.docker.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.docker.txt

COPY . .

RUN python docker_check.py

CMD ["python", "integra.py"]

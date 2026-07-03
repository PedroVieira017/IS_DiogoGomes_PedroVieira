FROM python:3.10-slim

# Evitar a geração de ficheiros .pyc e garantir outputs em tempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependências do sistema necessárias para compilações básicas (se houver)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar as dependências de Python
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# Copiar todo o código do repositório
COPY . .

# Comando padrão
CMD ["python", "LogicRAG/driving_agent.py"]

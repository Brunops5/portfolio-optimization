# Dockerfile — TCC: Otimização de Portfólios de Criptoativos
# Usa Python 3.12 slim como base para imagem enxuta

FROM python:3.12-slim

# Evita buffers no stdout/stderr (útil para logs em containers)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Diretório de trabalho
WORKDIR /app

# Instala dependências do sistema (compilação de pacotes numéricos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências Python primeiro (cache de layers)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do projeto
COPY . .

# Cria diretórios de saída (garante permissões corretas no container)
RUN mkdir -p data/raw data/processed results/metrics results/weights \
    reports/figures reports/tables

# Ponto de entrada padrão: executa o pipeline completo
CMD ["python", "main.py"]

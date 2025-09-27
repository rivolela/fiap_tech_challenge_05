# Dockerfile para a API de Scoring da Decision com suporte a monitoramento

FROM python:3.9-slim

WORKDIR /opt/render/project/src

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar apenas requirements primeiro para aproveitar cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código do projeto
COPY src/ ./src/
COPY models/ ./models/
# Criar diretórios de dados necessários
RUN mkdir -p ./data/processed/ ./data/metrics/ ./data/logs/
# Copiar scripts
COPY scripts/ ./scripts/

# Criar diretório para logs e métricas de monitoramento
RUN mkdir -p /opt/render/project/logs /opt/render/project/metrics

# Configurar variáveis de ambiente
ENV PYTHONPATH=/opt/render/project/src
ENV PORT=8000
ENV LOG_LEVEL=INFO
ENV LOG_FILE=/opt/render/project/logs/api_logs.log

# Expor a porta para a API e para o dashboard de monitoramento
EXPOSE 8000 8501

# Health check para garantir que a aplicação está funcionando
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1

# Criar script de inicialização com configurações otimizadas
RUN echo '#!/bin/bash\n\
# Configurar limites de recursos\n\
export PYTHONHASHSEED=random\n\
export PYTHONDONTWRITEBYTECODE=1\n\
export PYTHONUNBUFFERED=1\n\
# Garantir que diretórios de logs existem e têm permissões adequadas\n\
echo "Preparando diretórios de logs..."\n\
mkdir -p /opt/render/project/logs /opt/render/project/metrics /opt/render/project/src/data/logs /opt/render/project/src/logs logs\n\
chmod -R 777 /opt/render/project/logs /opt/render/project/src/logs /opt/render/project/src/data/logs logs\n\
touch /opt/render/project/logs/api_logs.log\n\
chmod 666 /opt/render/project/logs/api_logs.log\n\
echo "LOG_FILE definido como: $LOG_FILE"\n\
# Verificar se o diretório do LOG_FILE existe\n\
LOG_DIR=$(dirname "$LOG_FILE")\n\
mkdir -p "$LOG_DIR"\n\
chmod -R 777 "$LOG_DIR"\n\
touch "$LOG_FILE"\n\
chmod 666 "$LOG_FILE"\n\
echo "Teste de escrita em $(date)" >> "$LOG_FILE"\n\
echo "Verificando escrita em $LOG_FILE:"\n\
cat "$LOG_FILE"\n\
# Iniciar API com configurações otimizadas\n\
gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT \
--timeout 120 --graceful-timeout 60 --keep-alive 5 --max-requests 1000 \
--max-requests-jitter 50 --worker-tmp-dir /dev/shm \
--log-level debug src.api.scoring_api:app\n\
' > /opt/render/project/src/start.sh && chmod +x /opt/render/project/src/start.sh

# Iniciar API com Gunicorn para produção
CMD ["/opt/render/project/src/start.sh"]
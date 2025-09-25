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
# Criar diretório de dados processados (pode estar vazio ou não existir)
RUN mkdir -p ./data/processed/
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

# Criar script de inicialização
RUN echo '#!/bin/bash\n\
# Iniciar API\n\
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT src.api.scoring_api:app\n\
' > /opt/render/project/src/start.sh && chmod +x /opt/render/project/src/start.sh

# Iniciar API com Gunicorn para produção
CMD ["/opt/render/project/src/start.sh"]
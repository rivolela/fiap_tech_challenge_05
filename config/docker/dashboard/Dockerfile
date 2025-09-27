FROM python:3.9-slim

WORKDIR /opt/render/project

# Instalar dependências necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir streamlit matplotlib seaborn plotly

# Criar diretórios necessários
RUN mkdir -p /opt/render/project/data/monitoring/tmp
RUN mkdir -p /opt/render/project/logs

# Copiar o código
COPY src/ ./src/
COPY models/ ./models/

# Expor a porta do Streamlit
EXPOSE 8501

# Copiar o script de inicialização de métricas
COPY scripts/initialize_metrics.sh ./scripts/

# Comando para iniciar o dashboard com inicialização de métricas
CMD ["/bin/bash", "-c", "bash ./scripts/initialize_metrics.sh && streamlit run src/dashboard/dashboard.py --server.port=8501 --server.address=0.0.0.0"]
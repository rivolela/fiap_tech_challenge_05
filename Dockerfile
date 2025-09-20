# Dockerfile para a API de Scoring da Decision

FROM python:3.9-slim

WORKDIR /opt/render/project/src

# Instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn gunicorn

# Copiar todo o conteúdo do projeto
COPY . .

# Configurar variáveis de ambiente
ENV PYTHONPATH=/opt/render/project/src
ENV PORT=8000

# Expor a porta para a API
EXPOSE 8000

# Iniciar API com Gunicorn para produção
# Gunicorn é usado como servidor WSGI para produção
# Uvicorn funciona como worker para lidar com ASGI
CMD cd /opt/render/project/src && PYTHONPATH=/opt/render/project/src gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT src.api.scoring_api:app
#!/bin/bash
# Script para iniciar a API do modelo de scoring

set -e

echo "=== DECISION SCORING API ==="

# Verificar se o modelo existe
if [ ! -f "models/scoring_model.pkl" ]; then
    echo "❌ Modelo não encontrado em models/scoring_model.pkl"
    echo "Execute o treinamento primeiro com: ./scripts/run_pipeline.sh"
    exit 1
fi

# Verificar se as dependências estão instaladas
pip show fastapi uvicorn > /dev/null || {
    echo "📦 Instalando dependências necessárias..."
    pip install fastapi uvicorn
}

# Garantir que o TextBlob está instalado para a funcionalidade de LLM
pip install -r requirements.txt

# Executar a API
echo "🚀 Iniciando a API na porta 8000..."
uvicorn src.api.scoring_api:app --host 0.0.0.0 --port 8000

# Notas de uso:
# Para iniciar com recarga automática (desenvolvimento):
#   uvicorn src.api.scoring_api:app --host 0.0.0.0 --port 8000 --reload
# 
# Para iniciar com múltiplos workers em produção:
#   gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 src.api.scoring_api:app
#
# Documentação da API disponível em:
#   http://localhost:8000/docs
#
# Uso básico da API:
#   curl -X POST "http://localhost:8000/predict/" \
#        -H "X-API-Key: your-api-key" \
#        -H "Content-Type: application/json" \
#        -d '{
#              "idade": 30,
#              "experiencia": 5,
#              "educacao": "ensino_superior",
#              "area_formacao": "tecnologia",
#              "habilidades": ["python", "machine_learning"],
#              "vaga_titulo": "Desenvolvedor Python",
#              "vaga_area": "tecnologia",
#              "vaga_senioridade": "pleno"
#            }'
#
# Healthcheck:
#   curl "http://localhost:8000/health"

# Implementação de recursos:
# ✅ Autenticação via API key
# ✅ Validação de dados de entrada
# ✅ Documentação interativa (Swagger)
# ✅ Endpoints para predição individual e em lote
# ✅ Rota de verificação de saúde
# ✅ Métricas de performance
# ✅ Logging de requisições
# ✅ Tratamento de erros com mensagens descritivas
# ✅ Inferência de valores faltantes
# ✅ Geração de comentários via LLM
#
# O sistema também é compatível com Docker, sendo possível
# executar a API com:
#   docker build -t decision-scoring-api .
#   docker run -p 8000:8000 decision-scoring-api
#!/bin/bash
# Script para iniciar a API do modelo de scoring

set -e

echo "=== DECISION SCORING API ==="

# Verificar se o modelo existe
if [ ! -f "models/scoring_model.pkl" ]; then
    echo "❌ Modelo não encontrado em models/scoring_model.pkl"
    echo "Execute o treinamento primeiro com: ./run_pipeline.sh"
    exit 1
fi

# Verificar se as dependências estão instaladas
pip show fastapi uvicorn > /dev/null || {
    echo "📦 Instalando dependências necessárias..."
    pip install fastapi uvicorn
}

# Definir variáveis de ambiente
export PYTHONPATH=$(pwd)

# Iniciar o servidor API
echo "🚀 Iniciando API no endereço http://localhost:8000"
echo "📚 Documentação disponível em http://localhost:8000/docs"
echo ""
echo "📌 API Keys disponíveis para teste:"
echo "   - your-api-key (admin)"
echo "   - test-api-key (read-only)"
echo ""
echo "📌 Para autenticar, você pode usar:"
echo "   - Parâmetro de consulta: ?api_key=your-api-key"
echo "   - Cabeçalho HTTP: X-API-Key: your-api-key"
echo ""
echo "Pressione Ctrl+C para encerrar o servidor"

# Iniciar o servidor
uvicorn src.api.scoring_api:app --host 0.0.0.0 --port 8000 --reload
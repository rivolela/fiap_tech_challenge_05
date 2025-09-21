#!/bin/bash

# Script para implantação local rápida da API de scoring

echo "🚀 Iniciando implantação local da API de scoring"

# Verificar se o ambiente virtual existe
if [ ! -d ".venv" ]; then
    echo "📦 Criando ambiente virtual..."
    python3 -m venv .venv
fi

# Ativar ambiente virtual
echo "🔌 Ativando ambiente virtual..."
source .venv/bin/activate

# Instalar dependências
echo "📚 Instalando dependências..."
pip install -r requirements.txt

# Verificar existência do modelo
echo "🔍 Verificando modelo..."
if [ ! -f "models/scoring_model.pkl" ]; then
    echo "⚠️ Modelo não encontrado. Executando treinamento rápido..."
    python src/models/train_simple.py --quick
else
    echo "✅ Modelo encontrado em models/scoring_model.pkl"
fi

# Iniciar API
echo "🚀 Iniciando API..."
./scripts/start_api.sh

echo "✅ Implantação concluída"
echo "📊 Acesse a documentação da API em http://localhost:8000/docs"
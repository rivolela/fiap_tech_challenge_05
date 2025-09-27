#!/bin/bash
# Script para implantação local rápida da API de scoring
# Este script faz o setup básico e inicia a API localmente

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

# Verificar se os modelos estão presentes
if [ ! -f "models/scoring_model.pkl" ] || [ ! -f "models/feature_scaler.pkl" ]; then
    echo "⚠️ Modelos não encontrados. Treinando modelos..."
    python src/models/train_simple.py
fi

# Iniciar a API
echo "🌐 Iniciando a API de scoring..."
uvicorn src.api.scoring_api:app --reload --host 0.0.0.0 --port 8000

# Desativar ambiente virtual ao finalizar
deactivate
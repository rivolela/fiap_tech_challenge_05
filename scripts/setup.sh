#!/bin/bash
# Setup script para o projeto FIAP Tech Challenge 05

echo "=== CONFIGURAÇÃO DO AMBIENTE PYTHON ==="
echo "Criando ambiente virtual..."

# Criar ambiente virtual se não existir
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Ambiente virtual criado"
else
    echo "✅ Ambiente virtual já existe"
fi

# Ativar ambiente virtual
source .venv/bin/activate
echo "✅ Ambiente virtual ativado"

# Atualizar pip
pip install --upgrade pip

# Instalar dependências do projeto
echo "Instalando dependências..."
pip install -r requirements.txt

echo "✅ Dependências instaladas"

# Verificar instalação
echo "Verificando instalação..."
python scripts/check_env.py

echo "=== CONFIGURAÇÃO CONCLUÍDA ==="
echo "Para ativar o ambiente, execute: source .venv/bin/activate"
echo "Para executar o pipeline, use: ./scripts/run_pipeline.sh"
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

# Instalar dependências
echo "📦 Instalando dependências..."
pip install -r requirements.txt

echo ""
echo "🎉 Setup concluído!"
echo ""
echo "Para ativar o ambiente virtual, execute:"
echo "source .venv/bin/activate"
echo ""
echo "Para verificar a instalação, execute:"
echo "python check_env.py"
echo ""
echo "Para iniciar o Jupyter Notebook, execute:"
echo "jupyter notebook"

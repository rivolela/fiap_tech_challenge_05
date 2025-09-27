#!/bin/bash

# Script para executar os testes unitários

echo "🧪 Executando testes unitários..."

# Instalar dependências de desenvolvimento se necessário
echo "📦 Instalando dependências de testes..."
pip3 install -r requirements-dev.txt

# Verificar se pytest-cov está instalado
if ! pip3 show pytest-cov > /dev/null 2>&1; then
    echo "📦 Instalando pytest-cov..."
    pip3 install pytest-cov
fi

# Verificar se o diretório de testes existe
if [ ! -d "tests/unit" ]; then
    echo "❌ Diretório de testes não encontrado!"
    exit 1
fi

# Executar os testes com cobertura
echo "🚀 Iniciando execução dos testes com relatório de cobertura..."
python3 -m pytest tests/unit -v --cov=src --cov-report=term --cov-report=html

# Verificar se os testes foram executados com sucesso
if [ $? -eq 0 ]; then
    echo "✅ Todos os testes foram executados com sucesso!"
    REPORT_PATH="$(pwd)/htmlcov/index.html"
    echo "📊 O relatório de cobertura foi gerado em: $REPORT_PATH"
    # Tentar abrir o relatório
    if [ -f "$REPORT_PATH" ]; then
        echo "🔍 Abrindo o relatório de cobertura..."
        open "$REPORT_PATH"
    else
        echo "⚠️ O arquivo do relatório não foi encontrado em: $REPORT_PATH"
        echo "Procurando o relatório em todo o projeto..."
        find "$(pwd)" -name "index.html" | grep -i cov
    fi
else
    echo "❌ Alguns testes falharam. Verifique os erros acima."
    exit 1
fi
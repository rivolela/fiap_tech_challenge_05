#!/bin/bash

# Script para iniciar o dashboard de monitoramento de métricas
# Este script inicia o servidor Streamlit para exibir o dashboard de métricas

# Obter o diretório atual (que deve ser o diretório raiz do projeto)
PROJECT_DIR=$(pwd)

# Verificar se o Streamlit está instalado
if ! command -v streamlit &> /dev/null
then
    echo "Streamlit não está instalado. Instalando agora..."
    pip install streamlit pandas matplotlib seaborn
fi

# Adicionar o diretório do projeto ao PYTHONPATH
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

echo "Iniciando dashboard de monitoramento..."
streamlit run src/dashboard/dashboard.py
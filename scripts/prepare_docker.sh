#!/bin/bash

# Script para preparar a estrutura de diretórios para o Docker
# Autor: GitHub Copilot

echo "🔧 Preparando estrutura de diretórios para Docker..."

# Criar diretórios necessários
mkdir -p data/processed
mkdir -p data/monitoring/drift_reports
mkdir -p data/insights
mkdir -p logs
mkdir -p mlruns

# Criar arquivos README nos diretórios vazios
echo "# Diretório de dados processados

Este diretório contém dados processados utilizados pelo modelo de ML." > data/processed/README.md

echo "# Diretório de logs

Este diretório armazena logs da API e outros componentes." > logs/README.md

echo "# Diretório de métricas de monitoramento

Este diretório armazena métricas de monitoramento do modelo." > data/monitoring/README.md

echo "# Diretório de relatórios de drift

Este diretório armazena relatórios de análise de drift do modelo." > data/monitoring/drift_reports/README.md

echo "# Diretório de insights

Este diretório armazena análises e visualizações do modelo." > data/insights/README.md

# Aplicar permissões
chmod -R 755 data logs mlruns

echo "✅ Estrutura de diretórios preparada com sucesso!"
echo "📦 Agora você pode construir as imagens Docker com 'docker compose build'"
#!/bin/bash
# Script para iniciar a API com as configurações corretas de log

echo "🚀 Iniciando API de Scoring com configuração de logs corrigida..."

# Garantir que o diretório de logs existe e tem as permissões corretas
mkdir -p logs
chmod -R 777 logs

# Verificar se o arquivo de log existe e definir permissões
if [ -f "logs/api_logs.log" ]; then
    chmod 666 "logs/api_logs.log"
fi

# Iniciar a API com variáveis de ambiente configuradas
LOG_FILE=logs/api_logs.log \
CLASSIFICATION_THRESHOLD=0.25 \
python -m src.api.scoring_api

# Este script nunca chegará aqui a menos que a API termine
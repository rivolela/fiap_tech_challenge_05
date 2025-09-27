#!/bin/bash
# debug_logs.sh - Verifica e corrige problemas com os logs no Docker

# Criar diretório de logs localmente com permissões adequadas
mkdir -p logs
chmod -R 777 logs
touch logs/api_logs.log
chmod 666 logs/api_logs.log

echo "Diretório de logs local preparado com permissões adequadas"

# Verificar configuração atual
echo "Verificando configuração atual do docker-compose.yml..."
grep -A5 "volumes:" docker-compose.yml

# Executar o contêiner com comando para verificar os logs
echo -e "\nVerificando os logs dentro do contêiner..."
docker compose exec scoring-api bash -c "ls -la /opt/render/project/logs/ && echo 'LOG_FILE = ' \$LOG_FILE"

# Verificar se os logs estão sendo gerados
echo -e "\nTestando escrita nos logs..."
docker compose exec scoring-api bash -c "echo 'Teste de log em \$(date)' >> \$LOG_FILE && cat \$LOG_FILE | tail -5"

echo -e "\nVerificando logs locais..."
ls -la logs/
cat logs/api_logs.log | tail -5
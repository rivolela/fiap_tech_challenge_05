#!/bin/bash
# Script para corrigir permissões de diretórios no ambiente de produção

# Garantir que todos os diretórios necessários existem e têm permissões corretas
mkdir -p /opt/render/project/src/data/processed/ 
mkdir -p /opt/render/project/src/data/metrics/ 
mkdir -p /opt/render/project/src/data/logs/
mkdir -p /opt/render/project/logs
mkdir -p /opt/render/project/metrics
mkdir -p /opt/render/project/src/logs
mkdir -p logs

# Garantir permissões de escrita nos diretórios de logs
chmod -R 777 /opt/render/project/logs
chmod -R 777 /opt/render/project/src/logs
chmod -R 777 logs
chmod -R 777 /opt/render/project/src/data/logs/

# Criar arquivo de log vazio se não existir
touch /opt/render/project/logs/api_logs.log
chmod 666 /opt/render/project/logs/api_logs.log

echo "Diretórios de logs e permissões corrigidos"
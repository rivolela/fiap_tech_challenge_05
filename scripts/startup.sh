#!/bin/bash
# Script para configuração dos limites de memória e recursos para uso em ambiente com recursos limitados

# Limitar uso de memória do Python (em bytes, aqui está definido para 512MB)
export PYTHONMEMORY=536870912

# Configurar GC do Python para ser mais agressivo em ambientes com memória limitada
export PYTHONGC=1

# Definir limite de profundidade de recursão
export PYTHONRECURSIONLIMIT=1000

# Configurar variáveis para o Gunicorn e Uvicorn
export GUNICORN_CMD_ARGS="--timeout 120 --graceful-timeout 60 --keep-alive 5 --log-level debug --max-requests 1000 --max-requests-jitter 50 --workers 2"

# Iniciar a aplicação com configurações otimizadas
exec "$@"
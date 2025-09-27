#!/bin/bash

# Script para verificar drift do modelo e enviar alertas
# Este script deve ser executado periodicamente (por exemplo, via cron) para monitorar o drift

# Adicionar o diretório do projeto ao PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Verificando drift do modelo e alertas..."
python -c "from src.monitoring.alert_system import check_all_alerts; check_all_alerts()"

echo "Verificação completa!"
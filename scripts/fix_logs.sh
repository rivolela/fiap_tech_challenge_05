#!/bin/bash
# Script para diagnosticar e corrigir problemas com logs da API

echo "🔍 Diagnosticando problema com os logs da API..."
cd "$(dirname "$0")/.."

# Verificar a existência dos diretórios de logs
echo -e "\n📁 Verificando diretórios de logs:"
for log_dir in "logs" "data/logs"; do
  if [ -d "$log_dir" ]; then
    echo "  ✅ $log_dir existe"
    ls -la "$log_dir"
  else
    echo "  ❌ $log_dir não existe"
    echo "    👉 Criando diretório..."
    mkdir -p "$log_dir"
    echo "    ✅ Diretório $log_dir criado"
  fi
done

# Verificar arquivo de logs
LOG_FILE="logs/api_logs.log"
echo -e "\n📄 Verificando arquivo de logs ($LOG_FILE):"
if [ -f "$LOG_FILE" ]; then
  echo "  ✅ Arquivo de log existe"
  echo "  📊 Permissões atuais:"
  ls -la "$LOG_FILE"
  
  echo "  📋 Últimas 5 linhas do log:"
  tail -n 5 "$LOG_FILE"
else
  echo "  ❌ Arquivo de log não existe"
  echo "    👉 Criando arquivo de log vazio..."
  touch "$LOG_FILE"
  echo "    ✅ Arquivo de log criado"
fi

# Corrigir permissões
echo -e "\n🔑 Corrigindo permissões:"
echo "  👉 Definindo permissões de escrita para todos os usuários nos diretórios de logs..."
chmod -R 777 logs
chmod -R 777 data/logs 2>/dev/null

echo "  👉 Definindo permissões de escrita para todos os usuários no arquivo de log..."
chmod 666 "$LOG_FILE"

echo "  ✅ Permissões corrigidas"
ls -la "$LOG_FILE"

# Testar a escrita no log
echo -e "\n✏️ Testando a escrita no arquivo de log:"
echo "[$(date)] Teste de escrita realizado pelo script de diagnóstico" >> "$LOG_FILE"
if [ $? -eq 0 ]; then
  echo "  ✅ Escrita bem-sucedida no arquivo de log"
  echo "  📋 Últimas 5 linhas do log:"
  tail -n 5 "$LOG_FILE"
else
  echo "  ❌ Falha ao escrever no arquivo de log"
fi

echo -e "\n🔄 Recomendações para iniciar a API:"
echo "  1. Use o comando: python -m src.api.scoring_api"
echo "  2. Ou configure explicitamente o arquivo de log: LOG_FILE=logs/api_logs.log python -m src.api.scoring_api"
echo -e "\n✅ Diagnóstico concluído. Problemas de permissão devem estar resolvidos."
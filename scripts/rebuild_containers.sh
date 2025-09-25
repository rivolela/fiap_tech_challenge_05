#!/bin/bash
# Script para reconstruir os contêineres Docker com as novas configurações

echo "🐳 Reconstruindo contêineres Docker com as novas configurações..."

# Garantir que os diretórios necessários existam
mkdir -p data/monitoring/tmp
mkdir -p logs

# Inicializar métricas se necessário
if [ ! -f data/monitoring/model_metrics.json ]; then
  echo "📊 Inicializando métricas..."
  bash scripts/initialize_metrics.sh
else
  echo "✅ Arquivos de métricas já existem."
fi

# Parar contêineres em execução
echo "🛑 Parando contêineres em execução..."
docker-compose down

# Reconstruir os contêineres
echo "🔨 Reconstruindo contêineres..."
docker-compose up --build -d

# Verificar status
echo "🔍 Verificando status dos contêineres..."
docker-compose ps

echo "
✅ Configuração concluída!

📋 Serviços disponíveis:
- API de Scoring: http://localhost:8000
- Dashboard de Monitoramento: http://localhost:8502
- MLflow UI: http://localhost:5001

📘 Para mais informações, consulte docs/guia_monitoramento.md
"
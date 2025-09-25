#!/bin/bash
# Script para inicializar as métricas de monitoramento para o modelo de scoring

echo "📊 Inicializando métricas de monitoramento..."

# Verificar ambiente e ativar virtualenv se existir
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo "✅ Ambiente virtual ativado."
fi

# Criar diretório para armazenamento de métricas
mkdir -p data/monitoring/tmp
echo "✅ Diretórios de métricas criados."

# Criar arquivo inicial de métricas
cat > data/monitoring/model_metrics.json << 'EOF'
{
  "model_info": {
    "creation_date": "2025-09-25T10:00:00",
    "model_version": "1.0.0",
    "baseline_metrics": {
      "accuracy": 0.97,
      "precision": 0.84,
      "recall": 0.70,
      "f1_score": 0.76,
      "roc_auc": 0.98
    }
  },
  "metrics_history": [
    {
      "timestamp": "2025-09-25T10:00:00",
      "metrics": {
        "accuracy": 0.97,
        "precision": 0.84,
        "recall": 0.70,
        "f1_score": 0.76,
        "roc_auc": 0.98
      }
    }
  ]
}
EOF

echo "✅ Arquivo de métricas inicial criado em data/monitoring/model_metrics.json"

# Criar arquivo de log de predições vazio
cat > data/monitoring/predictions_log.csv << 'EOF'
timestamp,candidate_id,prediction,prediction_probability,features,segment
EOF

echo "✅ Arquivo de log de predições criado em data/monitoring/predictions_log.csv"

# Criar arquivo de drift inicial
cat > data/monitoring/drift_reports.json << 'EOF'
{
  "latest_report": {
    "timestamp": "2025-09-25T10:00:00",
    "overall_drift": 0.02,
    "feature_drift": {
      "idade": 0.01,
      "experiencia_anos": 0.02,
      "num_empregos_anteriores": 0.03,
      "tempo_empresa_atual": 0.01,
      "distancia_empresa": 0.02
    },
    "performance_metrics": {
      "accuracy": 0.97,
      "precision": 0.84,
      "recall": 0.70,
      "f1_score": 0.76
    }
  },
  "drift_history": [
    {
      "timestamp": "2025-09-25T10:00:00",
      "overall_drift": 0.02
    }
  ]
}
EOF

echo "✅ Arquivo de relatórios de drift criado em data/monitoring/drift_reports.json"

echo "📊 Inicialização de métricas concluída!"
echo "Você pode agora iniciar o dashboard com: streamlit run src/dashboard/dashboard.py"
"""
Função para atualizar os relatórios de drift de forma simulada
Criada para uso com o dashboard quando os endpoints da API não estão disponíveis
"""

import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

def update_drift_report(drift_data: Dict[str, Any], timestamp: Optional[str] = None):
    """
    Atualiza o arquivo de relatórios de drift com novos dados.
    
    Args:
        drift_data: Dicionário contendo dados de drift, incluindo 'overall_drift', 
                  'feature_drift' e 'performance_metrics'
        timestamp: Timestamp ISO opcional (se não fornecido, usa o tempo atual)
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().isoformat()
    
    # Caminho do arquivo de relatórios de drift
    drift_reports_path = Path("data/monitoring/drift_reports.json")
    
    # Verificar se o arquivo existe
    if not drift_reports_path.exists():
        # Criar um relatório inicial se o arquivo não existir
        initial_report = {
            "latest_report": {
                "timestamp": timestamp,
                "overall_drift": drift_data.get("overall_drift", 0.0),
                "feature_drift": drift_data.get("feature_drift", {}),
                "performance_metrics": drift_data.get("performance_metrics", {}),
                "n_samples_analyzed": drift_data.get("n_samples_analyzed", 150),  # Valor simulado padrão
                "features_analyzed": drift_data.get("features_analyzed", len(drift_data.get("feature_drift", {})))
            },
            "drift_history": []
        }
        with open(drift_reports_path, 'w') as f:
            json.dump(initial_report, f, indent=2)
    
    # Carregar relatórios existentes
    with open(drift_reports_path, 'r') as f:
        drift_reports = json.load(f)
    
    # Atualizar o relatório mais recente
    drift_reports["latest_report"] = {
        "timestamp": timestamp,
        "overall_drift": drift_data.get("overall_drift", 0.0),
        "feature_drift": drift_data.get("feature_drift", {}),
        "performance_metrics": drift_data.get("performance_metrics", {}),
        "n_samples_analyzed": drift_data.get("n_samples_analyzed", 150),  # Valor simulado padrão
        "features_analyzed": drift_data.get("features_analyzed", len(drift_data.get("feature_drift", {})))
    }
    
    # Adicionar ao histórico de drift
    drift_history_entry = {
        "timestamp": timestamp,
        "overall_drift": drift_data.get("overall_drift", 0.0)
    }
    drift_reports["drift_history"].append(drift_history_entry)
    
    # Salvar o arquivo atualizado
    with open(drift_reports_path, 'w') as f:
        json.dump(drift_reports, f, indent=2)

# Também vamos adicionar funções para simular a adição de entradas no log de predições
def add_simulated_predictions(num_predictions=5):
    """
    Adiciona predições simuladas ao log de predições.
    
    Args:
        num_predictions: Número de predições simuladas a serem adicionadas
    """
    import pandas as pd
    import numpy as np
    import random
    import json
    
    # Caminho do arquivo de log de predições
    predictions_log_path = Path("data/monitoring/predictions_log.csv")
    
    # Verificar se o arquivo existe
    if not predictions_log_path.exists():
        # Criar um arquivo vazio se não existir
        pd.DataFrame(columns=[
            'timestamp', 'candidate_id', 'prediction', 'prediction_probability',
            'features', 'segment'
        ]).to_csv(predictions_log_path, index=False)
    
    # Carregar log existente
    predictions_df = pd.read_csv(predictions_log_path)
    
    # Gerar predições simuladas
    new_predictions = []
    segments = ["tech", "marketing", "sales", "operations", "finance"]
    
    for i in range(num_predictions):
        timestamp = datetime.datetime.now().isoformat()
        candidate_id = f"CAND-{random.randint(10000, 99999)}"
        prediction = random.choice([0, 1])
        probability = random.uniform(0.55, 0.98) if prediction == 1 else random.uniform(0.02, 0.45)
        
        # Gerar características simuladas
        features = {
            "idade": random.randint(22, 55),
            "experiencia_anos": random.randint(1, 20),
            "num_empregos_anteriores": random.randint(1, 5),
            "tempo_empresa_atual": random.randint(0, 10),
            "distancia_empresa": round(random.uniform(1, 50), 1)
        }
        
        new_predictions.append({
            "timestamp": timestamp,
            "candidate_id": candidate_id,
            "prediction": prediction,
            "prediction_probability": round(probability, 4),
            "features": json.dumps(features),
            "segment": random.choice(segments)
        })
    
    # Adicionar novas predições ao log
    new_df = pd.DataFrame(new_predictions)
    updated_df = pd.concat([predictions_df, new_df], ignore_index=True)
    
    # Salvar o log atualizado
    updated_df.to_csv(predictions_log_path, index=False)
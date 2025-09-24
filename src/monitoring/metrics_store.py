"""
metrics_store.py - Sistema de armazenamento e gerenciamento de métricas do modelo

Este módulo fornece funções para armazenar, recuperar e analisar métricas de desempenho
do modelo de scoring da Decision. Ele mantém um histórico de métricas para detectar
mudanças de comportamento e drift do modelo ao longo do tempo.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

# Definir o caminho para o armazenamento de métricas
METRICS_DIR = Path("data/monitoring")
METRICS_FILE = METRICS_DIR / "model_metrics.json"
PREDICTIONS_LOG = METRICS_DIR / "predictions_log.csv"

def initialize_metrics_store():
    """Inicializa o armazenamento de métricas, criando diretórios e arquivos se necessário."""
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # Inicializar arquivo de métricas se não existir
    if not METRICS_FILE.exists():
        with open(METRICS_FILE, 'w') as f:
            json.dump({
                "model_info": {
                    "creation_date": datetime.datetime.now().isoformat(),
                    "model_version": "1.0.0",
                    "baseline_metrics": {}
                },
                "metrics_history": []
            }, f, indent=2)
    
    # Inicializar log de predições se não existir
    if not PREDICTIONS_LOG.exists():
        pd.DataFrame(columns=[
            'timestamp', 'candidate_id', 'prediction', 'prediction_probability',
            'features', 'segment'
        ]).to_csv(PREDICTIONS_LOG, index=False)

def get_metrics_history() -> Dict[str, Any]:
    """Recupera o histórico completo de métricas."""
    try:
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        initialize_metrics_store()
        return get_metrics_history()

def save_model_metrics(metrics: Dict[str, float], timestamp: Optional[str] = None):
    """
    Salva métricas do modelo no histórico.
    
    Args:
        metrics: Dicionário contendo as métricas do modelo
        timestamp: Timestamp ISO opcional (se não fornecido, usa o tempo atual)
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().isoformat()
    
    metrics_data = get_metrics_history()
    
    # Adicionar novas métricas ao histórico
    metrics_entry = {
        "timestamp": timestamp,
        "metrics": metrics
    }
    
    metrics_data["metrics_history"].append(metrics_entry)
    
    # Salvar o arquivo atualizado
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics_data, f, indent=2)

def log_prediction(
    candidate_id: str, 
    prediction: int, 
    prediction_probability: float,
    features: Dict[str, Any],
    segment: Optional[str] = None
):
    """
    Registra uma predição do modelo para análise posterior.
    
    Args:
        candidate_id: Identificador único do candidato
        prediction: Predição do modelo (0 ou 1)
        prediction_probability: Probabilidade da predição
        features: Características usadas para fazer a predição
        segment: Segmento do candidato (opcional)
    """
    now = datetime.datetime.now().isoformat()
    
    # Converter features para string JSON para armazenamento
    features_json = json.dumps(features)
    
    # Carregar o CSV atual, adicionar nova linha e salvar
    try:
        df = pd.read_csv(PREDICTIONS_LOG)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame(columns=[
            'timestamp', 'candidate_id', 'prediction', 'prediction_probability',
            'features', 'segment'
        ])
    
    new_row = pd.DataFrame([{
        'timestamp': now,
        'candidate_id': candidate_id,
        'prediction': prediction,
        'prediction_probability': prediction_probability,
        'features': features_json,
        'segment': segment if segment else 'default'
    }])
    
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(PREDICTIONS_LOG, index=False)

def get_recent_predictions(days: int = 30) -> pd.DataFrame:
    """
    Recupera as predições mais recentes para análise.
    
    Args:
        days: Número de dias para olhar para trás
        
    Returns:
        DataFrame com as predições recentes
    """
    try:
        df = pd.read_csv(PREDICTIONS_LOG)
        if len(df) == 0:
            return pd.DataFrame()
        
        # Converter timestamp para datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filtrar por data
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        recent_df = df[df['timestamp'] >= cutoff_date]
        
        # Converter features de string JSON para dicionários
        def parse_json(json_str):
            try:
                return json.loads(json_str)
            except:
                return {}
        
        recent_df['features_dict'] = recent_df['features'].apply(parse_json)
        
        return recent_df
    
    except FileNotFoundError:
        return pd.DataFrame()

def set_baseline_metrics(metrics: Dict[str, float]):
    """Define as métricas de linha de base para comparação futura."""
    metrics_data = get_metrics_history()
    metrics_data["model_info"]["baseline_metrics"] = metrics
    
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics_data, f, indent=2)

def get_baseline_metrics() -> Dict[str, float]:
    """Recupera as métricas de linha de base do modelo."""
    metrics_data = get_metrics_history()
    return metrics_data["model_info"]["baseline_metrics"]

def compare_with_baseline(current_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Compara as métricas atuais com a linha de base e retorna as diferenças.
    
    Args:
        current_metrics: Métricas atuais do modelo
        
    Returns:
        Dicionário com as diferenças entre as métricas atuais e a linha de base
    """
    baseline = get_baseline_metrics()
    
    if not baseline:
        return {k: 0.0 for k in current_metrics.keys()}
    
    return {k: current_metrics.get(k, 0) - baseline.get(k, 0) for k in set(current_metrics) | set(baseline)}
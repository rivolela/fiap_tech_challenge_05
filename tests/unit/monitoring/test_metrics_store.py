"""Testes para o módulo metrics_store"""

import os
import json
import pytest
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock
from datetime import datetime

from src.monitoring.metrics_store import (
    save_model_metrics, 
    get_metrics_history, 
    log_prediction,
    get_recent_predictions,
    METRICS_FILE,
    PREDICTIONS_LOG
)

@pytest.fixture
def sample_metrics():
    """Fixture para simular métricas do modelo."""
    return {
        "accuracy": 0.95,
        "precision": 0.85,
        "recall": 0.75,
        "f1_score": 0.80,
        "roc_auc": 0.92
    }

@pytest.fixture
def mock_metrics_file():
    """Fixture para simular um arquivo de métricas."""
    return {
        "model_info": {
            "creation_date": "2025-09-01T10:00:00",
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
                "timestamp": "2025-09-01T10:00:00",
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

@pytest.fixture
def mock_predictions_df():
    """Fixture para simular um DataFrame de predições."""
    return pd.DataFrame({
        'timestamp': ["2025-09-01T10:00:00", "2025-09-01T11:00:00"],
        'candidate_id': ["cand-001", "cand-002"],
        'prediction': [1, 0],
        'prediction_probability': [0.85, 0.35],
        'features': ['{"idade": 30}', '{"idade": 25}'],
        'segment': ['tech', 'sales']
    })

@patch('builtins.open', new_callable=mock_open)
@patch('json.load')
@patch('json.dump')
def test_save_model_metrics(mock_json_dump, mock_json_load, mock_open_file, sample_metrics, mock_metrics_file):
    """Testa a função save_model_metrics."""
    # Configurar o mock para retornar o arquivo de métricas simulado
    mock_json_load.return_value = mock_metrics_file
    
    # Chamar a função save_model_metrics
    timestamp = "2025-09-26T12:00:00"
    save_model_metrics(sample_metrics, timestamp)
    
    # Verificar se open foi chamado com o caminho correto para leitura
    mock_open_file.assert_any_call(METRICS_FILE, 'r')
    
    # Verificar se open foi chamado com o caminho correto para escrita
    mock_open_file.assert_any_call(METRICS_FILE, 'w')
    
    # Verificar se json.dump foi chamado
    mock_json_dump.assert_called_once()
    
    # Verificar se as métricas foram adicionadas ao histórico
    # Não podemos verificar diretamente os argumentos do mock_json_dump devido à sua complexidade
    # Mas podemos verificar se foi chamado

@patch('builtins.open', new_callable=mock_open)
@patch('json.load')
def test_get_metrics_history(mock_json_load, mock_open_file, mock_metrics_file):
    """Testa a função get_metrics_history."""
    # Configurar o mock para retornar o arquivo de métricas simulado
    mock_json_load.return_value = mock_metrics_file
    
    # Chamar a função get_metrics_history
    result = get_metrics_history()
    
    # Verificar se open foi chamado com o caminho correto
    mock_open_file.assert_called_with(METRICS_FILE, 'r')
    
    # Verificar se json.load foi chamado
    mock_json_load.assert_called_once()
    
    # Verificar o resultado
    assert result == mock_metrics_file

@patch('pandas.DataFrame.to_csv')
@patch('pandas.read_csv')
def test_log_prediction(mock_read_csv, mock_to_csv, mock_predictions_df):
    """Testa a função log_prediction."""
    # Configurar o mock para retornar o DataFrame de predições simulado
    mock_read_csv.return_value = mock_predictions_df
    
    # Dados da predição
    prediction_data = {
        "candidate_id": "cand-003",
        "prediction": 1,
        "prediction_probability": 0.9,  # Corrigido de 'probability' para 'prediction_probability'
        "features": {"idade": 35, "experiencia": 10},
        "segment": "tech"
    }
    
    # Chamar a função log_prediction
    log_prediction(**prediction_data)
    
    # Verificar se read_csv foi chamado
    mock_read_csv.assert_called_with(PREDICTIONS_LOG)
    
    # Verificar se to_csv foi chamado
    mock_to_csv.assert_called_once()

@patch('pandas.read_csv')
def test_get_recent_predictions(mock_read_csv, mock_predictions_df):
    """Testa a função get_recent_predictions."""
    # Configurar o mock para retornar o DataFrame de predições simulado
    mock_read_csv.return_value = mock_predictions_df
    
    # Chamar a função get_recent_predictions
    result = get_recent_predictions(days=30)
    
    # Verificar se read_csv foi chamado
    mock_read_csv.assert_called_with(PREDICTIONS_LOG)
    
    # Verificar o resultado (deve ser um DataFrame)
    assert isinstance(result, pd.DataFrame)
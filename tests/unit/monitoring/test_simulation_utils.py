"""Testes para o módulo simulation_utils"""

import os
import json
import pytest
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

from src.monitoring.simulation_utils import (
    update_drift_report,
    add_simulated_predictions
)

@pytest.fixture
def mock_drift_data():
    """Fixture para simular dados de drift."""
    return {
        "overall_drift": 0.03,
        "feature_drift": {
            "idade": 0.01,
            "experiencia_anos": 0.02,
            "num_empregos_anteriores": 0.03,
            "tempo_empresa_atual": 0.01,
            "distancia_empresa": 0.02
        },
        "performance_metrics": {
            "accuracy": 0.95,
            "precision": 0.85,
            "recall": 0.75,
            "f1_score": 0.80
        },
        "n_samples_analyzed": 150,
        "features_analyzed": 5
    }

@pytest.fixture
def mock_drift_report():
    """Fixture para simular um relatório de drift."""
    return {
        "latest_report": {
            "timestamp": "2025-09-26T10:00:00",
            "overall_drift": 0.02,
            "feature_drift": {
                "idade": 0.01,
                "experiencia_anos": 0.02,
                "num_empregos_anteriores": 0.03,
                "tempo_empresa_atual": 0.01,
                "distancia_empresa": 0.02
            },
            "performance_metrics": {
                "accuracy": 0.95,
                "precision": 0.85,
                "recall": 0.75,
                "f1_score": 0.80
            },
            "n_samples_analyzed": 150,
            "features_analyzed": 5
        },
        "drift_history": [
            {
                "timestamp": "2025-09-26T10:00:00",
                "overall_drift": 0.02
            }
        ]
    }

@patch('pathlib.Path.exists')
@patch('builtins.open', new_callable=mock_open)
@patch('json.load')
@patch('json.dump')
def test_update_drift_report_existing_file(
    mock_json_dump, 
    mock_json_load, 
    mock_open_file, 
    mock_exists,
    mock_drift_data, 
    mock_drift_report
):
    """Testa a função update_drift_report quando o arquivo já existe."""
    # Configurar os mocks
    mock_exists.return_value = True
    mock_json_load.return_value = mock_drift_report
    
    # Chamar a função update_drift_report
    timestamp = "2025-09-26T12:00:00"
    update_drift_report(mock_drift_data, timestamp)
    
    # Verificar se Path.exists foi chamado
    mock_exists.assert_called_once()
    
    # Verificar se open foi chamado duas vezes (leitura e escrita)
    assert mock_open_file.call_count == 2
    
    # Verificar se json.load foi chamado
    mock_json_load.assert_called_once()
    
    # Verificar se json.dump foi chamado
    mock_json_dump.assert_called_once()

@patch('pathlib.Path.exists')
@patch('builtins.open', new_callable=mock_open)
@patch('json.dump')
def test_update_drift_report_new_file(
    mock_json_dump, 
    mock_open_file, 
    mock_exists,
    mock_drift_data
):
    """Testa a função update_drift_report quando o arquivo não existe."""
    # Configurar os mocks
    mock_exists.return_value = False
    
    # Preparar um relatório inicial vazio mas válido
    initial_report = {
        "latest_report": {},
        "drift_history": []
    }
    
    # Configurar o mock_open_file para retornar o relatório inicial quando lido
    file_handle = mock_open_file.return_value
    file_handle.read.return_value = json.dumps(initial_report)
    
    # Chamar a função update_drift_report
    timestamp = "2025-09-26T12:00:00"
    update_drift_report(mock_drift_data, timestamp)
    
    # Verificar se Path.exists foi chamado
    mock_exists.assert_called_once()
    
    # Verificar se open foi chamado
    assert mock_open_file.call_count >= 1
    
    # Verificar se json.dump foi chamado
    assert mock_json_dump.call_count >= 1
    
    # Verificar se o relatório tem a estrutura esperada
    called_args = mock_json_dump.call_args[0][0]  # O primeiro argumento do primeiro call
    assert "latest_report" in called_args
    assert "drift_history" in called_args

import datetime

def test_add_simulated_predictions():
    """Testa a função add_simulated_predictions."""
    # Use patch para evitar a criação real de arquivos
    with patch('src.monitoring.simulation_utils.Path') as mock_path:
        # Configurar o comportamento dos mocks
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.exists.return_value = True
        
        # Criar DataFrames reais ao invés de mocks para evitar o erro InvalidSpecError
        existing_df = pd.DataFrame({
            'timestamp': ["2025-09-26T10:00:00"],
            'candidate_id': ["cand-001"],
            'prediction': [1],
            'prediction_probability': [0.8],
            'features': ['{"idade": 30}'],
            'segment': ['tech']
        })
        
        combined_df = pd.DataFrame()  # DataFrame vazio que será preenchido por concat
        
        # Configurar mocks para funções externas
        with patch('pandas.read_csv', return_value=existing_df):
            with patch('pandas.concat', return_value=combined_df):
                with patch('datetime.datetime') as mock_datetime:
                    with patch('random.randint', return_value=12345):
                        with patch('random.uniform', return_value=0.75):
                            with patch('random.choice', side_effect=["tech", 1, 0, "sales"]):
                                # Fixar datetime
                                fixed_datetime = datetime.datetime(2025, 9, 26, 12, 0, 0)
                                mock_datetime.now.return_value = fixed_datetime
                                
                                # Chamar a função com mock para to_csv
                                with patch.object(combined_df, 'to_csv') as mock_to_csv:
                                    add_simulated_predictions(num_predictions=2)
                                    
                                    # Verificar se Path foi chamado corretamente
                                    mock_path.assert_called_once()
                                    mock_path_instance.exists.assert_called_once()
                                    
                                    # Verificar se to_csv foi chamado
                                    mock_to_csv.assert_called_once()

@patch('src.monitoring.simulation_utils.Path')
@patch('pandas.DataFrame')
@patch('pandas.read_csv')
@patch('pandas.concat')
def test_add_simulated_predictions_new_file(mock_concat, mock_read_csv, mock_dataframe, mock_path):
    """Testa a função add_simulated_predictions quando o arquivo não existe."""
    # Configurar o comportamento dos mocks
    mock_path_instance = MagicMock()
    mock_path.return_value = mock_path_instance
    mock_path_instance.exists.return_value = False
    
    # Criar DataFrames reais para os mocks
    empty_df = pd.DataFrame()
    mock_dataframe.return_value = empty_df
    
    new_df = pd.DataFrame({
        'timestamp': ["2025-09-26T10:00:00"],
        'candidate_id': ["cand-001"],
        'prediction': [1],
        'prediction_probability': [0.8],
        'features': ['{"idade": 30}'],
        'segment': ['tech']
    })
    mock_concat.return_value = new_df
    
    # Mock para to_csv
    to_csv_mock = MagicMock()
    empty_df.to_csv = to_csv_mock
    new_df.to_csv = to_csv_mock
    
    # Mock de funções para datetime e valores aleatórios
    with patch('datetime.datetime') as mock_datetime:
        with patch('random.randint') as mock_randint:
            with patch('random.uniform') as mock_uniform:
                with patch('random.choice') as mock_choice:
                    # Configurar retornos para valores aleatórios
                    mock_datetime.now.return_value = datetime.datetime(2025, 9, 26, 12, 0, 0)
                    mock_randint.return_value = 12345
                    mock_uniform.return_value = 0.75
                    mock_choice.side_effect = ["tech", 1, "tech", 0]
                    
                    # Chamar a função
                    add_simulated_predictions(num_predictions=2)
                    
                    # Verificar chamadas
                    mock_path.assert_called_once()
                    mock_path_instance.exists.assert_called_once()
                    assert to_csv_mock.call_count >= 1
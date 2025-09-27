"""Testes para o módulo model_loader"""

import os
import pickle
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open, MagicMock

from src.api.model_loader import (
    load_model, 
    get_feature_list, 
    predict, 
    check_sklearn_version
)

@pytest.fixture
def mock_model():
    """Fixture para simular um modelo treinado."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    model.predict.return_value = np.array([1])
    model.feature_names_in_ = ["idade", "experiencia", "educacao"]
    return model

@pytest.fixture
def sample_input():
    """Fixture para simular dados de entrada."""
    return {
        "idade": 30,
        "experiencia": 5,
        "educacao": "ensino_superior",
        "area_formacao": "tecnologia",
        "habilidades": ["python", "sql", "aws"]
    }

@patch('pickle.load')
@patch('builtins.open', new_callable=mock_open)
def test_load_model(mock_open_file, mock_pickle_load, mock_model):
    """Testa a função load_model."""
    # Configurar o mock para retornar o modelo quando pickle.load for chamado
    mock_pickle_load.return_value = mock_model
    
    # Chamar a função load_model
    model = load_model()
    
    # Verificar se open foi chamado com o caminho correto
    mock_open_file.assert_called_with('models/scoring_model.pkl', 'rb')
    
    # Verificar se pickle.load foi chamado
    mock_pickle_load.assert_called_once()
    
    # Verificar se o modelo foi retornado corretamente
    assert model == mock_model

def test_get_feature_list():
    """Testa a função get_feature_list."""
    # Caso simples: verifica se retorna uma lista
    features = get_feature_list()
    assert isinstance(features, list)
    assert len(features) > 0

@patch('src.api.model_loader.load_model')
def test_predict(mock_load_model, mock_model, sample_input):
    """Testa a função predict."""
    # Configurar o mock para retornar o modelo simulado
    mock_load_model.return_value = mock_model
    
    # Criar um DataFrame simulado
    df = pd.DataFrame([list(sample_input.values())], columns=list(sample_input.keys()))
    
    # Chamar a função predict que agora retorna um dicionário
    result = predict(df)
    
    # Verificar se load_model foi chamado
    mock_load_model.assert_called_once()
    
    # Verificar se o modelo.predict_proba foi chamado
    mock_model.predict_proba.assert_called_once()
    
    # Verificar os resultados - agora usando o formato de dicionário
    assert result["prediction"] == 1
    assert result["probability"] == 0.7
    assert "recommendation" in result
    assert "comment" in result

@patch('builtins.print')
@patch('sklearn.__version__', '1.7.1')  # Simula a versão instalada
def test_check_sklearn_version(mock_print):
    """Testa a função check_sklearn_version."""
    # A função não retorna nada, apenas exibe informações
    check_sklearn_version()
    
    # Verificar que print foi chamado
    assert mock_print.called
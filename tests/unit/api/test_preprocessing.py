"""Testes para o módulo preprocessing"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.api.preprocessing import preprocess_input, preprocess_batch

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

@pytest.fixture
def sample_batch_input():
    """Fixture para simular dados de entrada em lote."""
    return {
        "candidates": [
            {
                "idade": 30,
                "experiencia": 5,
                "educacao": "ensino_superior",
                "area_formacao": "tecnologia",
                "habilidades": ["python", "sql", "aws"]
            },
            {
                "idade": 25,
                "experiencia": 2,
                "educacao": "ensino_medio",
                "area_formacao": "administracao",
                "habilidades": ["excel", "word"]
            }
        ]
    }

@patch('src.api.model_loader.get_feature_list')
def test_preprocess_input(mock_get_feature_list, sample_input):
    """Testa a função preprocess_input."""
    # Configurar o mock para retornar uma lista de features
    mock_get_feature_list.return_value = ["idade", "experiencia"]
    
    # Chamar a função preprocess_input
    result = preprocess_input(sample_input)
    
    # Verificar se o resultado é um DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Verificar se o DataFrame tem apenas uma linha
    assert len(result) == 1
    
    # Verificar se a idade está presente
    assert "idade" in result.columns
    assert result["idade"].iloc[0] == 30
    
    # Verificar se a experiência está presente
    assert "experiencia" in result.columns
    assert result["experiencia"].iloc[0] == 5
    
    # Verificar se get_feature_list foi chamado
    mock_get_feature_list.assert_called_once()

@patch('src.api.preprocessing.preprocess_input')
def test_preprocess_batch(mock_preprocess_input, sample_batch_input):
    """Testa a função preprocess_batch."""
    # Configurar o mock para retornar um DataFrame simulado
    mock_df1 = pd.DataFrame([[30, 5]], columns=["idade", "experiencia"])
    mock_df2 = pd.DataFrame([[25, 2]], columns=["idade", "experiencia"])
    mock_preprocess_input.side_effect = [mock_df1, mock_df2]
    
    # Chamar a função preprocess_batch
    # O formato esperado pela função é uma lista de dicionários
    result = preprocess_batch(sample_batch_input["candidates"])
    
    # Verificar se o resultado é um DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Verificar se o DataFrame tem duas linhas
    assert len(result) == 2
    
    # Verificar se preprocess_input foi chamado duas vezes
    assert mock_preprocess_input.call_count == 2
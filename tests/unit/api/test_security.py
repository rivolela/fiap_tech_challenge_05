"""Testes para o módulo security"""

import os
import pytest
import json
import datetime
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from fastapi.requests import Request

from src.api.security import (
    verify_api_key, 
    init_api_keys, 
    hash_api_key,
    get_api_keys_from_env,
    check_rate_limit
)

@pytest.fixture
def mock_request():
    """Fixture para simular um objeto Request."""
    request = MagicMock(spec=Request)
    client = MagicMock()
    client.host = "127.0.0.1"
    request.client = client
    request.url = MagicMock()
    request.url.path = "/test-path"
    return request

@pytest.fixture
def mock_api_keys():
    """Fixture para simular chaves de API."""
    return {
        "hashed-admin-key": "admin",
        "hashed-readonly-key": "read-only"
    }

@patch('src.api.security.hash_api_key')
@patch('src.api.security.check_rate_limit')
@patch('src.api.security.API_KEYS')
async def test_verify_api_key(mock_api_keys, mock_check_rate_limit, mock_hash_api_key, mock_request):
    """Testa a função verify_api_key."""
    # Configurar os mocks
    mock_hash_api_key.return_value = "hashed-admin-key"
    mock_api_keys.__getitem__.return_value = "admin"
    mock_api_keys.__contains__.return_value = True
    mock_check_rate_limit.return_value = True
    
    # Chamar a função verify_api_key
    role = await verify_api_key(mock_request, api_key="test-api-key", x_api_key=None)
    
    # Verificar o resultado
    assert role == "admin"
    
    # Verificar se hash_api_key foi chamado com a chave correta
    mock_hash_api_key.assert_called_with("test-api-key")
    
    # Verificar se check_rate_limit foi chamado corretamente
    mock_check_rate_limit.assert_called_with("test-api-key", "admin")

@patch('src.api.security.hash_api_key')
@patch('src.api.security.check_rate_limit')
@patch('src.api.security.API_KEYS')
async def test_verify_api_key_with_header(mock_api_keys, mock_check_rate_limit, mock_hash_api_key, mock_request):
    """Testa a função verify_api_key quando a chave vem do cabeçalho."""
    # Configurar os mocks
    mock_hash_api_key.return_value = "hashed-readonly-key"
    mock_api_keys.__getitem__.return_value = "read-only"
    mock_api_keys.__contains__.return_value = True
    mock_check_rate_limit.return_value = True
    
    # Chamar a função verify_api_key
    role = await verify_api_key(mock_request, api_key=None, x_api_key="header-api-key")
    
    # Verificar o resultado
    assert role == "read-only"
    
    # Verificar se hash_api_key foi chamado com a chave correta
    mock_hash_api_key.assert_called_with("header-api-key")

@patch('src.api.security.hash_api_key')
@patch('src.api.security.API_KEYS')
async def test_verify_api_key_invalid(mock_api_keys, mock_hash_api_key, mock_request):
    """Testa a função verify_api_key com chave inválida."""
    # Configurar o mock para retornar chave inválida
    mock_hash_api_key.return_value = "invalid-hash"
    mock_api_keys.__contains__.return_value = False
    
    # Verificar se uma exceção é lançada
    with pytest.raises(HTTPException) as excinfo:
        await verify_api_key(mock_request, api_key="invalid-key", x_api_key=None)
    
    # Verificar se a exceção tem o status code correto
    assert excinfo.value.status_code == 401

@patch('src.api.security.hash_api_key')
@patch('src.api.security.check_rate_limit')
@patch('src.api.security.API_KEYS')
async def test_verify_api_key_rate_limit_exceeded(mock_api_keys, mock_check_rate_limit, mock_hash_api_key, mock_request):
    """Testa a função verify_api_key quando o rate limit é excedido."""
    # Configurar os mocks
    mock_hash_api_key.return_value = "hashed-admin-key"
    mock_api_keys.__getitem__.return_value = "admin"
    mock_api_keys.__contains__.return_value = True
    mock_check_rate_limit.return_value = False
    
    # Verificar se uma exceção é lançada
    with pytest.raises(HTTPException) as excinfo:
        await verify_api_key(mock_request, api_key="test-api-key", x_api_key=None)
    
    # Verificar se a exceção tem o status code correto
    assert excinfo.value.status_code == 429

async def test_verify_api_key_no_key(mock_request):
    """Testa a função verify_api_key quando nenhuma chave é fornecida."""
    # Verificar se uma exceção é lançada
    with pytest.raises(HTTPException) as excinfo:
        await verify_api_key(mock_request, api_key=None, x_api_key=None)
    
    # Verificar se a exceção tem o status code correto
    assert excinfo.value.status_code == 401

@patch('json.loads')
@patch('os.environ.get')
def test_get_api_keys_from_env(mock_env_get, mock_json_loads):
    """Testa a função get_api_keys_from_env."""
    # Configurar os mocks
    mock_env_get.return_value = '[{"key": "admin-key", "role": "admin"}, {"key": "readonly-key", "role": "read-only"}]'
    mock_json_loads.return_value = [
        {"key": "admin-key", "role": "admin"},
        {"key": "readonly-key", "role": "read-only"}
    ]
    
    # Mock para hash_api_key
    with patch('src.api.security.hash_api_key', side_effect=["hashed-admin-key", "hashed-readonly-key"]):
        # Chamar a função get_api_keys_from_env
        result = get_api_keys_from_env()
        
        # Verificar o resultado
        assert "hashed-admin-key" in result
        assert result["hashed-admin-key"] == "admin"
        assert "hashed-readonly-key" in result
        assert result["hashed-readonly-key"] == "read-only"

@patch('src.api.security.get_api_keys_from_env')
def test_init_api_keys(mock_get_api_keys):
    """Testa a função init_api_keys."""
    # Configurar o mock
    mock_api_keys = {
        "hashed-admin-key": "admin",
        "hashed-readonly-key": "read-only"
    }
    mock_get_api_keys.return_value = mock_api_keys
    
    # Chamar a função init_api_keys com um mock para API_KEYS
    with patch('src.api.security.API_KEYS', {}) as mock_global_api_keys:
        init_api_keys()
        
        # Verificar se a variável global foi atualizada
        assert mock_get_api_keys.called
        # Note: Como não podemos facilmente verificar a variável global,
        # verificamos apenas se a função foi chamada

@patch('os.environ.get')
def test_hash_api_key(mock_env_get):
    """Testa a função hash_api_key."""
    # Configurar o mock
    mock_env_get.return_value = "test-salt"
    
    # Chamar a função hash_api_key
    hashed = hash_api_key("test-key")
    
    # Verificar se o resultado é uma string
    assert isinstance(hashed, str)
    
    # Verificar se o hash não é igual à chave original
    assert hashed != "test-key"
    
    # Verificar se a mesma chave produz o mesmo hash
    hashed2 = hash_api_key("test-key")
    assert hashed == hashed2

@patch('src.api.security.clean_old_request_counts')
@patch('src.api.security.datetime')
@patch('src.api.security.hash_api_key')
@patch('src.api.security.REQUEST_COUNTS')
@patch('src.api.security.REQUEST_TIMESTAMPS')
@patch('src.api.security.RATE_LIMIT_MAX_REQUESTS')
def test_check_rate_limit_first_request(
    mock_rate_limits, mock_timestamps, mock_counts, 
    mock_hash, mock_datetime, mock_clean
):
    """Testa a função check_rate_limit para primeira requisição."""
    # Configurar os mocks
    mock_hash.return_value = "hashed-key"
    mock_rate_limits.get.return_value = 100
    mock_counts.__contains__.return_value = False
    current_time = MagicMock()
    mock_datetime.now.return_value = current_time
    
    # Chamar a função check_rate_limit
    result = check_rate_limit("test-key", "admin")
    
    # Verificar o resultado
    assert result is True
    
    # Verificar se os contadores foram atualizados
    mock_counts.__setitem__.assert_called_with("hashed-key", 1)
    mock_timestamps.__setitem__.assert_called_with("hashed-key", current_time)

def test_check_rate_limit_under_limit():
    """Testa a função check_rate_limit para requisições dentro do limite."""
    # Usar um escopo de função para isolar os efeitos colaterais
    class MockDict(dict):
        """Um dicionário mockado que suporta incremento"""
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
    
    # Criar um dicionário real que podemos modificar
    mock_counts_dict = MockDict({"hashed-key": 50})
    
    # Patch módulos necessários para o teste
    with patch('src.api.security.clean_old_request_counts'):
        with patch('src.api.security.hash_api_key', return_value="hashed-key"):
            with patch('src.api.security.RATE_LIMIT_MAX_REQUESTS', {"admin": 100}):
                with patch('src.api.security.REQUEST_COUNTS', mock_counts_dict):
                    now = datetime.datetime.now()  # Usar um datetime real
                    with patch('src.api.security.REQUEST_TIMESTAMPS', {"hashed-key": now}):
                        with patch('src.api.security.datetime') as mock_datetime:
                            mock_datetime.now.return_value = now
                            # Chamar a função check_rate_limit
                            result = check_rate_limit("test-key", "admin")
    
    # Verificar o resultado
    assert result is True

def test_check_rate_limit_over_limit():
    """Testa a função check_rate_limit para requisições acima do limite."""
    # Usar um escopo de função para isolar os efeitos colaterais
    class MockDict(dict):
        """Um dicionário mockado que suporta incremento"""
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
    
    # Criar um dicionário real que podemos modificar
    mock_counts_dict = MockDict({"hashed-key": 101})
    
    # Patch módulos necessários para o teste
    with patch('src.api.security.clean_old_request_counts'):
        with patch('src.api.security.hash_api_key', return_value="hashed-key"):
            with patch('src.api.security.RATE_LIMIT_MAX_REQUESTS', {"admin": 100}):
                with patch('src.api.security.REQUEST_COUNTS', mock_counts_dict):
                    now = datetime.datetime.now()  # Usar um datetime real
                    with patch('src.api.security.REQUEST_TIMESTAMPS', {"hashed-key": now}):
                        with patch('src.api.security.datetime') as mock_datetime:
                            mock_datetime.now.return_value = now
                            # Chamar a função check_rate_limit
                            result = check_rate_limit("test-key", "admin")
    
    # Verificar o resultado
    assert result is False
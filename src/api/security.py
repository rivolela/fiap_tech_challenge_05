"""Módulo para gerenciar a segurança e autenticação da API"""

import os
import logging
import hmac
import hashlib
import base64
from typing import Optional, Dict
from datetime import datetime, timedelta
from fastapi import Depends, Header, HTTPException, status, Query, Request
from functools import lru_cache
from dotenv import load_dotenv

# Configurar o logger
logger = logging.getLogger("decision-api.security")

# Carregar variáveis de ambiente
load_dotenv()

# Chaves de API (serão inicializadas pela função init_api_keys)
# Formato: {"chave_api_hashed": "role"}
API_KEYS: Dict[str, str] = {
    # Valores padrão, serão substituídos pela função init_api_keys
    # Admin key
    "526ad77089d41f0b24c9c4dbdb1d861173a0b7d12b5da3148ca86c3ae56cd75c": "admin",  # your-api-key (hashed)
    "074b4cc16ac5a29907bc44f4abf13e5158363416ce10d2cc77fb12252d242ffa": "admin",  # fiap-api-key (hashed)
    # Read-only key
    "ceb1aaa0d16c8851422baa230eed00417def9c13cb7dfff0c55f257a77dcae9b": "read-only"  # test-api-key (hashed)
}

# Variável para controle de rate limiting (implementação simples para exemplo)
# Em produção, use Redis ou outro sistema distribuído
REQUEST_COUNTS = {}
REQUEST_TIMESTAMPS = {}

# Tempo de expiração do rate limit em segundos
RATE_LIMIT_WINDOW = 60  # 1 minuto
RATE_LIMIT_MAX_REQUESTS = {
    "admin": 100,  # 100 requisições por minuto
    "read-only": 30  # 30 requisições por minuto
}


def hash_api_key(api_key: str) -> str:
    """
    Gera um hash seguro para a API key.
    Usa SHA-256 com um salt armazenado em variável de ambiente.
    """
    salt = os.environ.get("API_KEY_SALT", "default-salt-change-in-production")
    
    # Criar um hash utilizando HMAC para segurança adicional
    h = hmac.new(
        salt.encode('utf-8'),
        api_key.encode('utf-8'),
        hashlib.sha256
    )
    return h.hexdigest()


def clean_old_request_counts():
    """Limpa contagens de requisições antigas para o rate limiting"""
    current_time = datetime.now()
    
    # Encontrar chaves que expiraram
    expired_keys = [
        key for key, timestamp in REQUEST_TIMESTAMPS.items()
        if (current_time - timestamp).total_seconds() > RATE_LIMIT_WINDOW
    ]
    
    # Remover chaves expiradas
    for key in expired_keys:
        if key in REQUEST_COUNTS:
            del REQUEST_COUNTS[key]
        if key in REQUEST_TIMESTAMPS:
            del REQUEST_TIMESTAMPS[key]


def check_rate_limit(api_key: str, role: str) -> bool:
    """
    Verifica se o cliente atingiu o limite de requisições.
    Retorna True se está dentro do limite, False caso contrário.
    """
    clean_old_request_counts()
    
    # Obter limite de requisições para o role
    max_requests = RATE_LIMIT_MAX_REQUESTS.get(role, 10)  # Padrão 10 req/min
    
    # Verificar se a chave está no dicionário
    current_time = datetime.now()
    hashed_key = hash_api_key(api_key)
    
    if hashed_key not in REQUEST_COUNTS:
        REQUEST_COUNTS[hashed_key] = 1
        REQUEST_TIMESTAMPS[hashed_key] = current_time
        return True
    
    # Verificar se o período de tempo expirou
    last_request = REQUEST_TIMESTAMPS[hashed_key]
    if (current_time - last_request).total_seconds() > RATE_LIMIT_WINDOW:
        # Reset contador para nova janela de tempo
        REQUEST_COUNTS[hashed_key] = 1
        REQUEST_TIMESTAMPS[hashed_key] = current_time
        return True
    
    # Incrementar contador e verificar limite
    REQUEST_COUNTS[hashed_key] += 1
    if REQUEST_COUNTS[hashed_key] > max_requests:
        return False
    
    return True


async def get_api_key_header(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[str]:
    """Extrai a API key do cabeçalho HTTP"""
    return x_api_key


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Query(None, description="API Key para autenticação"),
    x_api_key: Optional[str] = Depends(get_api_key_header)
) -> str:
    """
    Verifica se a API key fornecida é válida e gerencia o rate limiting.
    Retorna o role associado à API key.
    """
    # Obter a API key da query ou cabeçalho
    key = api_key or x_api_key
    
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key não fornecida. Forneça via parâmetro 'api_key' ou cabeçalho 'X-API-Key'"
        )
    
    # Remover espaços em branco e gerar hash
    key = key.strip() if key else None
    hashed_key = hash_api_key(key)
    
    # Verificar se a chave existe
    # Adicionar logs para debug
    logger.warning(f"API key recebida: {key}")
    logger.warning(f"Hash gerado: {hashed_key}")
    logger.warning(f"Chaves disponíveis: {list(API_KEYS.keys())}")
    
    if hashed_key not in API_KEYS:
        # Usando abordagem segura para evitar erro com None
        key_info = key[:4] + "..." + key[-4:] if key and len(key) > 8 else key
        logger.warning(f"Tentativa de acesso com API Key inválida: {key_info}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida"
        )
    
    role = API_KEYS[hashed_key]
    
    # Verificar rate limiting
    # Aqui sabemos que key não é None porque passamos pela validação anterior
    assert key is not None
    if not check_rate_limit(key, role):
        # Usando abordagem segura para evitar erro com None
        key_info = key[:4] + "..." + key[-4:] if key and len(key) > 8 else key
        logger.warning(f"Rate limit excedido para API Key: {key_info}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Limite de requisições excedido. Tente novamente em {RATE_LIMIT_WINDOW} segundos."
        )
    
    # Registrar uso da API (apenas para debug)
    client_ip = request.client.host if request.client else "unknown"
    logger.debug(f"API Key válida ({role}) usada de {client_ip} para {request.url.path}")
    
    return role


@lru_cache(maxsize=10)
def get_api_keys_from_env() -> Dict[str, str]:
    """
    Carrega as chaves de API de variáveis de ambiente.
    O formato da variável de ambiente deve ser uma lista de dicionários JSON.
    Exemplo: API_KEYS='[{"key": "chave1", "role": "admin"}, {"key": "chave2", "role": "read-only"}]'
    """
    import json
    import os
    
    try:
        # Tentar carregar do ambiente
        env_keys = os.environ.get("API_KEYS")
        if not env_keys:
            logger.warning("Variável de ambiente API_KEYS não encontrada. Usando chaves padrão.")
            return API_KEYS
        
        # Parsear JSON
        keys_data = json.loads(env_keys)
        
        # Converter para o formato interno
        result = {}
        for item in keys_data:
            if "key" in item and "role" in item:
                # Hashear a chave
                hashed_key = hash_api_key(item["key"])
                result[hashed_key] = item["role"]
        
        if not result:
            logger.warning("Nenhuma API Key válida encontrada no ambiente. Usando chaves padrão.")
            return API_KEYS
            
        logger.info(f"Carregadas {len(result)} chaves de API do ambiente.")
        return result
    
    except Exception as e:
        logger.error(f"Erro ao carregar API Keys do ambiente: {str(e)}. Usando chaves padrão.")
        return API_KEYS


def init_api_keys():
    """Inicializa as chaves de API do ambiente ou usa as padrão"""
    global API_KEYS
    API_KEYS = get_api_keys_from_env()
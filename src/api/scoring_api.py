"""API FastAPI para o sistema de scoring da Decision"""

from fastapi import FastAPI, HTTPException, Depends, Query, Header, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
import pandas as pd
import numpy as np
import os
import time
import datetime
import logging
import uuid
from typing import Dict, List, Optional, Any

# Importar módulos locais
from src.api.schemas import (
    CandidateRequest,
    BatchCandidateRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
    MetricsResponse
)
from src.api.model_loader import load_model, get_feature_list, check_sklearn_version
from src.api.preprocessing import preprocess_input, preprocess_batch

# Verificar a versão do scikit-learn
check_sklearn_version()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_logs.log")
    ]
)
logger = logging.getLogger("decision-api")

# Inicializar a API
app = FastAPI(
    title="Decision Scoring API",
    description="API para scoring de candidatos usando o modelo de machine learning da Decision",
    version="1.0.0",
)

# Configurar CORS para permitir requisições de origens específicas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar apenas origens confiáveis
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Métricas da API
API_START_TIME = time.time()
REQUEST_COUNT = 0
ERROR_COUNT = 0
LATENCY_SUM = 0.0

# Chaves de API válidas (em um sistema real, use um sistema de gerenciamento de secrets)
# Em produção, armazene em um banco de dados seguro ou serviço de gerenciamento de segredos
API_KEYS = {
    "your-api-key": "admin",
    "test-api-key": "read-only"
}

# Função para obter o cabeçalho X-API-Key
def get_api_key_header(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    return x_api_key

# Verificação de autenticação por API key
def verify_api_key(
    api_key: Optional[str] = Query(None, description="API Key para autenticação"),
    x_api_key: Optional[str] = Depends(get_api_key_header)
):
    """Verifica se a API key fornecida é válida, seja por query param ou header"""
    # Obter a API key do cabeçalho se não estiver na query
    key = api_key or x_api_key
    
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key não fornecida. Forneça via parâmetro 'api_key' ou cabeçalho 'X-API-Key'"
        )
    
    # Remover espaços em branco antes e depois da chave
    key = key.strip() if key else None
    
    if key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida"
        )
    return API_KEYS[key]

# Função para registrar métricas de API
def update_metrics(start_time: float, success: bool):
    """Atualiza métricas da API com base na requisição"""
    global REQUEST_COUNT, ERROR_COUNT, LATENCY_SUM
    
    REQUEST_COUNT += 1
    if not success:
        ERROR_COUNT += 1
    
    latency = time.time() - start_time
    LATENCY_SUM += latency

# Middleware para logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware para logging de requisições"""
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        process_time = time.time() - start_time
        logger.info(
            f"Request {request_id} completed: {response.status_code} "
            f"({process_time:.4f}s)"
        )
        
        update_metrics(start_time, response.status_code < 400)
        
        return response
    except Exception as e:
        logger.exception(f"Request {request_id} failed: {str(e)}")
        update_metrics(start_time, False)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Erro interno do servidor", "detail": str(e)}
        )


# Endpoints da API

@app.post(
    "/predict/", 
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    description="Realiza uma predição para um candidato em relação a uma vaga"
)
async def predict(
    candidate: CandidateRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Endpoint para predição individual de candidatos
    
    Este endpoint avalia a compatibilidade de um candidato com uma vaga específica.
    Você pode fornecer informações sobre a vaga usando os campos vaga_id, vaga_titulo, 
    vaga_area e vaga_senioridade para obter uma predição mais precisa.
    """
    try:
        logger.info(f"Processando predição para candidato: {candidate.dict()}")
        
        # Converter para o formato que o modelo espera
        input_data = candidate.dict()
        df = preprocess_input(input_data)
        
        # Carregar modelo e fazer a predição
        from src.api.model_loader import predict
        prediction_result = predict(df)
        
        logger.info(f"Predição concluída: {prediction_result}")
        
        # Preparar informações da vaga
        vaga_info = None
        match_score = None
        
        if any(candidate.dict().get(field) for field in ['vaga_id', 'vaga_titulo', 'vaga_area', 'vaga_senioridade']):
            vaga_info = {
                "id": candidate.vaga_id or "não especificado",
                "titulo": candidate.vaga_titulo or "não especificado",
                "area": candidate.vaga_area or "não especificada",
                "senioridade": candidate.vaga_senioridade or "não especificada"
            }
            
            # Calcular uma pontuação de compatibilidade
            # Aqui estamos usando uma simplificação, em produção isso seria mais sofisticado
            base_score = prediction_result["probability"]
            
            # Ajustar com base em fatores específicos da vaga
            match_score = base_score
            
            # Se existe correspondência direta entre área de formação e vaga, aumentar o score
            if 'match_area' in df.columns and df['match_area'].iloc[0] == 1:
                match_score = min(1.0, match_score * 1.2)  # Aumento de 20%, máximo de 1.0
        
        # Retornar resultado formatado
        return PredictionResponse(
            prediction=prediction_result["prediction"],
            probability=prediction_result["probability"],
            recommendation=prediction_result["recommendation"],
            comment=prediction_result.get("comment"),
            vaga_info=vaga_info,
            match_score=match_score
        )
    
    except Exception as e:
        logger.exception(f"Erro ao processar predição: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/predict/batch/", 
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    description="Realiza predições em lote para múltiplos candidatos"
)
async def predict_batch(
    request: BatchCandidateRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Endpoint para predição em lote de múltiplos candidatos
    """
    try:
        batch_size = len(request.candidates)
        logger.info(f"Processando predição em lote para {batch_size} candidatos")
        
        # Processar cada candidato
        results = []
        
        for candidate in request.candidates:
            input_data = candidate.dict()
            df = preprocess_input(input_data)
            
            # Fazer a predição
            from src.api.model_loader import predict
            prediction_result = predict(df)
            
            # Preparar informações da vaga
            vaga_info = None
            match_score = None
            
            if any(candidate.dict().get(field) for field in ['vaga_id', 'vaga_titulo', 'vaga_area', 'vaga_senioridade']):
                vaga_info = {
                    "id": candidate.vaga_id or "não especificado",
                    "titulo": candidate.vaga_titulo or "não especificado",
                    "area": candidate.vaga_area or "não especificada",
                    "senioridade": candidate.vaga_senioridade or "não especificada"
                }
                
                # Calcular uma pontuação de compatibilidade
                base_score = prediction_result["probability"]
                match_score = base_score
                
                # Se existe correspondência direta entre área de formação e vaga, aumentar o score
                if 'match_area' in df.columns and df['match_area'].iloc[0] == 1:
                    match_score = min(1.0, match_score * 1.2)  # Aumento de 20%, máximo de 1.0
            
            results.append(PredictionResponse(
                prediction=prediction_result["prediction"],
                probability=prediction_result["probability"],
                recommendation=prediction_result["recommendation"],
                comment=prediction_result.get("comment"),
                vaga_info=vaga_info,
                match_score=match_score
            ))
        
        logger.info(f"Predição em lote concluída para {batch_size} candidatos")
        
        # Retornar resultados
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        return BatchPredictionResponse(
            results=results,
            timestamp=timestamp
        )
    
    except Exception as e:
        logger.exception(f"Erro ao processar predição em lote: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get(
    "/health", 
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    description="Verifica o status de saúde da API"
)
@app.get(
    "/health/", 
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    description="Verifica o status de saúde da API"
)
async def health():
    """
    Endpoint para verificação de saúde da API
    """
    try:
        # Tentar carregar o modelo para verificar se está funcionando
        model = load_model()
        
        # Informações sobre o modelo
        model_info = {
            "model_type": type(model).__name__,
            "n_features": len(get_feature_list()),
            "features": get_feature_list()[:5] + ["..."] if len(get_feature_list()) > 5 else get_feature_list()
        }
        
        # Se o modelo tiver atributos adicionais, incluir
        if hasattr(model, "n_estimators"):
            model_info["n_estimators"] = model.n_estimators
            
        if hasattr(model, "n_classes_"):
            model_info["n_classes"] = model.n_classes_
        
        # Retornar informações de saúde
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            model_info=model_info
        )
    
    except Exception as e:
        logger.exception(f"Erro na verificação de saúde: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"API em estado não saudável: {str(e)}"
        )


@app.get(
    "/metrics/", 
    response_model=MetricsResponse,
    status_code=status.HTTP_200_OK,
    description="Retorna métricas de desempenho da API"
)
async def get_metrics(role: str = Depends(verify_api_key)):
    """
    Endpoint para obter métricas de desempenho da API
    Requer autenticação com API key
    """
    # Verificar se o usuário tem permissão de admin
    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permissão insuficiente para acessar métricas. Requer API key com nível 'admin'."
        )
    
    # Calcular métricas
    uptime_seconds = time.time() - API_START_TIME
    uptime_days = int(uptime_seconds / (60*60*24))
    uptime_hours = int((uptime_seconds % (60*60*24)) / (60*60))
    uptime_minutes = int((uptime_seconds % (60*60)) / 60)
    
    uptime_str = f"{uptime_days}d {uptime_hours}h {uptime_minutes}m"
    
    error_rate = 0.0
    avg_latency = 0.0
    
    if REQUEST_COUNT > 0:
        error_rate = (ERROR_COUNT / REQUEST_COUNT) * 100
        avg_latency = LATENCY_SUM / REQUEST_COUNT * 1000  # em milissegundos
    
    # Obter métricas do modelo (em um sistema real, estas seriam monitoradas)
    model_metrics = {
        "accuracy": 0.85,  # Exemplo, em produção use métricas reais
        "precision": 0.82,
        "recall": 0.87,
        "f1_score": 0.84
    }
    
    return MetricsResponse(
        uptime=uptime_str,
        request_count=REQUEST_COUNT,
        average_latency=round(avg_latency, 2),
        error_rate=round(error_rate, 2),
        model_metrics=model_metrics
    )


@app.get(
    "/", 
    status_code=status.HTTP_200_OK,
    description="Endpoint raiz da API"
)
async def root():
    """
    Endpoint raiz com informações básicas
    """
    return {
        "api": "Decision Scoring API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Personalizar a documentação OpenAPI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Decision Scoring API",
        version="1.0.0",
        description=(
            "API para predição de sucesso de candidatos usando o modelo "
            "de machine learning desenvolvido pela Decision."
            "\n\n"
            "## Autenticação\n"
            "Esta API requer autenticação via API Key. Você pode fornecer a API Key de duas formas:\n"
            "1. Via query parameter: `?api_key=your-api-key`\n" 
            "2. Via HTTP header: `X-API-Key: your-api-key`\n"
            "\n"
            "API Keys disponíveis:\n"
            "- `your-api-key`: Acesso de administrador (todos os endpoints)\n"
            "- `test-api-key`: Acesso somente leitura (endpoints básicos)\n"
            "\n"
            "## Endpoints\n"
            "- `/predict/`: Predição individual de candidatos\n"
            "- `/predict/batch/`: Predição em lote para múltiplos candidatos\n"
            "- `/health/`: Verificação de saúde da API\n"
            "- `/metrics/`: Métricas de desempenho (requer permissão de admin)\n"
        ),
        routes=app.routes,
    )
    
    # Adicionar informações adicionais
    openapi_schema["info"]["contact"] = {
        "name": "Suporte Decision",
        "email": "suporte@decision.tech",
        "url": "https://decision.tech/suporte",
    }
    
    openapi_schema["info"]["license"] = {
        "name": "Proprietary",
        "url": "https://decision.tech/termos",
    }
    
    # Adicionar tag para categorizar endpoints
    openapi_schema["tags"] = [
        {
            "name": "prediction",
            "description": "Endpoints para predição de candidatos",
        },
        {
            "name": "monitoring",
            "description": "Endpoints para monitoramento da API",
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
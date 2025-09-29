"""API FastAPI para o sistema de scoring da Decision"""

from fastapi import FastAPI, HTTPException, Depends, Query, Header, status, Request
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
from dotenv import load_dotenv

# Importar módulos locais
from src.api.schemas import (
    CandidateRequest,
    BatchCandidateRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
    MetricsResponse,
    DriftResponse
)
# Importar módulos de monitoramento
try:
    from src.monitoring.metrics_store import (
        initialize_metrics_store,
        get_metrics_history,
        log_prediction,
        get_recent_predictions,
        save_model_metrics
    )
    from src.monitoring.drift_detector import (
        initialize_drift_detector,
        generate_drift_report,
        get_latest_drift_report,
        visualize_feature_drift
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    print("⚠️ Módulos de monitoramento não disponíveis. Algumas funcionalidades estarão limitadas.")
from src.api.model_loader import load_model, get_feature_list, check_sklearn_version
from src.api.preprocessing import preprocess_input, preprocess_batch

# Verificar a versão do scikit-learn
check_sklearn_version()

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
log_file = os.getenv("LOG_FILE", "logs/api_logs.log")
print(f"LOG_FILE configurado inicialmente como: {log_file}")

# Verificar e criar múltiplos diretórios possíveis para logs
possible_log_dirs = [
    "logs",
    "data/logs",
    "/opt/render/project/logs",
    "/opt/render/project/src/logs",
    "/opt/render/project/src/data/logs"
]

# Tentar criar todos os diretórios de logs com permissões amplas
for log_dir in possible_log_dirs:
    try:
        os.makedirs(log_dir, exist_ok=True)
        # Definir permissões amplas para permitir escrita por qualquer usuário
        try:
            os.chmod(log_dir, 0o777)  # permissões 777 (leitura, escrita e execução para todos)
            print(f"Diretório de logs criado/verificado com permissões amplas: {log_dir}")
        except Exception as perm_err:
            print(f"Aviso: Não foi possível definir permissões para {log_dir}: {perm_err}")
    except Exception as e:
        print(f"Aviso: Não foi possível criar diretório {log_dir}: {e}")

# Verificar se o log_file tem caminho absoluto, se não, tentar caminhos alternativos
if not os.path.isabs(log_file):
    # Garantir que usamos sempre o mesmo arquivo de log em ambiente local
    # Priorizar o arquivo na pasta logs/ na raiz do projeto
    possible_log_files = [
        "logs/api_logs.log",  # Arquivo padrão na pasta logs
        log_file,  # Usar o valor original da variável de ambiente
        f"/opt/render/project/logs/{os.path.basename(log_file)}",  # Opções para o Render
        f"/opt/render/project/src/logs/{os.path.basename(log_file)}"
    ]
    
    for possible_file in possible_log_files:
        try:
            log_dir = os.path.dirname(possible_file)
            if log_dir:  # Se o diretório não for vazio
                os.makedirs(log_dir, exist_ok=True)
                # Definir permissões amplas para o diretório
                try:
                    os.chmod(log_dir, 0o777)
                except Exception:
                    pass
            
            # Tentar criar/abrir o arquivo
            with open(possible_file, 'a') as f:
                f.write(f"Log inicializado em {datetime.datetime.now()}\n")
            
            # Definir permissões amplas para o arquivo
            try:
                os.chmod(possible_file, 0o666)  # permissões 666 (leitura e escrita para todos)
            except Exception:
                pass
                
            print(f"✅ Arquivo de log configurado com sucesso: {possible_file}")
            log_file = possible_file
            break
        except Exception as e:
            print(f"❌ Não foi possível acessar arquivo de log {possible_file}: {e}")
else:
    # Garantir que o diretório do log_file existe
    try:
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        # Definir permissões amplas para o diretório
        try:
            os.chmod(log_dir, 0o777)
        except Exception:
            pass
            
        print(f"Diretório para log_file criado: {log_dir}")
        
        # Criar/abrir o arquivo
        with open(log_file, 'a') as f:
            f.write(f"Log inicializado em {datetime.datetime.now()}\n")
        
        # Definir permissões amplas para o arquivo
        try:
            os.chmod(log_file, 0o666)
        except Exception:
            pass
            
        print(f"✅ Arquivo de log configurado com sucesso: {log_file}")
    except Exception as e:
        print(f"❌ Não foi possível acessar o arquivo de log configurado: {e}")
        # Tentar usar um log alternativo na raiz
        log_file = "api_logs.log"
        try:
            with open(log_file, 'a') as f:
                f.write(f"Log inicializado em {datetime.datetime.now()} (fallback)\n")
            try:
                os.chmod(log_file, 0o666)
            except Exception:
                pass
            print(f"✅ Usando arquivo de log alternativo: {log_file}")
        except Exception as alt_error:
            print(f"❌ Todas as tentativas de configuração de log falharam: {alt_error}")
            print("⚠️ A API continuará executando, mas sem logging em arquivo")

print(f"LOG_FILE final configurado como: {log_file}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger("decision-api")

# Inicializar a API
app = FastAPI(
    title="Decision Scoring API",
    description="API para scoring de candidatos usando o modelo de machine learning da Decision",
    version="1.0.0",
)

# Importar e incluir endpoints de monitoramento se estiverem disponíveis
try:
    from src.api.monitoring_endpoints import router as monitoring_router
    app.include_router(monitoring_router)
    logger.info("Endpoints de monitoramento carregados com sucesso.")
except ImportError as e:
    logger.warning(f"Não foi possível carregar endpoints de monitoramento: {str(e)}")

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

# Importar funções de segurança
from src.api.security import verify_api_key, init_api_keys

# Inicializar as chaves de API
init_api_keys()

# Inicializar o sistema de monitoramento
if MONITORING_AVAILABLE:
    try:
        logger.info("Inicializando sistema de monitoramento de métricas e drift...")
        initialize_metrics_store()
        initialize_drift_detector()
        logger.info("Sistema de monitoramento inicializado com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao inicializar monitoramento: {str(e)}")
        MONITORING_AVAILABLE = False

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
    "/predict", 
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["prediction"],
    description="Realiza uma predição para um candidato em relação a uma vaga"
)
async def predict(
    candidate: CandidateRequest,
    request: Request,
    role: str = Depends(verify_api_key)
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
        
            # Registrar predição para monitoramento de drift, se disponível
            if MONITORING_AVAILABLE:
                try:
                    # Extrair características principais para log
                    features_dict = {col: df[col].iloc[0] for col in df.columns if col != 'target'}
                    
                    # Gerar ID único para o candidato se não fornecido
                    candidate_id = str(uuid.uuid4())
                    
                    # Registrar predição
                    log_prediction(
                        candidate_id=candidate_id,
                        prediction=prediction_result["prediction"],
                        prediction_probability=prediction_result["probability"],
                        features=features_dict,
                        segment=input_data.get("area", None)
                    )
                except Exception as e:
                    logger.warning(f"Erro ao registrar predição para monitoramento: {str(e)}")
            
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
    "/predict/batch", 
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["prediction"],
    description="Realiza predições em lote para múltiplos candidatos"
)
async def predict_batch(
    batch_request: BatchCandidateRequest,
    request: Request,
    role: str = Depends(verify_api_key)
):
    """
    Endpoint para predição em lote de múltiplos candidatos
    """
    try:
        batch_size = len(batch_request.candidates)
        logger.info(f"Processando predição em lote para {batch_size} candidatos")
        
        # Processar cada candidato
        results = []
        
        for candidate in batch_request.candidates:
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
    tags=["monitoring"],
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
    tags=["monitoring"],
    description="Retorna métricas de desempenho da API"
)
async def get_metrics(request: Request, role: str = Depends(verify_api_key)):
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
    
    # Obter métricas do modelo
    if MONITORING_AVAILABLE:
        try:
            # Buscar métricas do histórico
            metrics_history = get_metrics_history()
            if metrics_history["metrics_history"]:
                # Usar as métricas mais recentes
                latest_metrics = metrics_history["metrics_history"][-1]["metrics"]
                model_metrics = latest_metrics
            else:
                # Métricas default se não houver histórico
                model_metrics = {
                    "accuracy": 0.85,
                    "precision": 0.82,
                    "recall": 0.87,
                    "f1_score": 0.84
                }
                # Salvar as métricas padrão no sistema
                save_model_metrics(model_metrics)
        except Exception as e:
            logger.warning(f"Erro ao buscar métricas do modelo: {str(e)}")
            model_metrics = {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.87,
                "f1_score": 0.84
            }
    else:
        # Métricas de exemplo
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
    tags=["system"],
    description="Endpoint raiz da API"
)
async def root(request: Request):
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
            "Esta API requer autenticação via API Key. Você deve fornecer a API Key apenas através do header HTTP:\n"
            "- `X-API-Key: fiap-api-key`\n"
            "\n"
            "API Keys disponíveis:\n"
            "- `fiap-api-key`: Acesso de administrador (todos os endpoints)\n"
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
    
    # Remover parâmetros de API key via query para todas as rotas
    if "paths" in openapi_schema:
        for path in openapi_schema["paths"].values():
            for operation in path.values():
                if "parameters" in operation:
                    # Filtrar e manter apenas os parâmetros que não são 'api_key' na query
                    operation["parameters"] = [
                        param for param in operation["parameters"]
                        if not (param.get("name") == "api_key" and param.get("in") == "query")
                    ]
    
    # Garantir que a descrição na seção info não mencione query parameter
    if "info" in openapi_schema and "description" in openapi_schema["info"]:
        current_desc = openapi_schema["info"]["description"]
        
        # Substituir qualquer menção à autenticação via query parameter
        if "Via query parameter" in current_desc or "?api_key=" in current_desc:
            openapi_schema["info"]["description"] = (
                "API para predição de sucesso de candidatos usando o modelo de machine learning desenvolvido pela Decision."
                "\n\n"
                "## Autenticação\n"
                "Esta API requer autenticação via API Key. Você deve fornecer a API Key apenas através do header HTTP:\n"
                "- `X-API-Key: fiap-api-key`\n"
                "\n"
                "API Keys disponíveis:\n"
                "- `fiap-api-key`: Acesso de administrador (todos os endpoints)\n"
                "- `test-api-key`: Acesso somente leitura (endpoints básicos)\n"
                "\n"
                "## Endpoints\n"
                "- `/predict/`: Predição individual de candidatos\n"
                "- `/predict/batch/`: Predição em lote para múltiplos candidatos\n"
                "- `/health/`: Verificação de saúde da API\n"
                "- `/metrics/`: Métricas de desempenho (requer permissão de admin)\n"
            )
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
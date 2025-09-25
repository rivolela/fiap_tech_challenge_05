"""
monitoring_endpoints.py - Endpoints adicionais para monitoramento do modelo

Este módulo contém endpoints adicionais para monitorar o desempenho e drift do modelo.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, status, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
import os
import time
import datetime
import logging
import uuid
import io
import threading
import subprocess
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.api.schemas import DriftResponse
from src.api.security import verify_api_key

# Importar módulos de monitoramento
try:
    from src.monitoring.metrics_store import (
        get_metrics_history,
        get_recent_predictions,
        get_baseline_metrics,
        compare_with_baseline
    )
    from src.monitoring.drift_detector import (
        generate_drift_report,
        get_latest_drift_report,
        visualize_feature_drift
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# Configurar logging
logger = logging.getLogger("decision-api")

# Diretório para armazenar gráficos temporários
TMP_DIR = Path("data/monitoring/tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# Criar router para endpoints de monitoramento
router = APIRouter(
    prefix="/monitoring",
    tags=["monitoring"],
    dependencies=[Depends(verify_api_key)]
)

@router.get(
    "/drift",
    response_model=DriftResponse,
    status_code=status.HTTP_200_OK,
    description="Retorna análise de drift do modelo"
)
async def get_drift_analysis(
    request: Request, 
    role: str = Depends(verify_api_key),
    refresh: bool = Query(False, description="Se True, força uma nova análise de drift")
):
    """
    Endpoint para obter análise de drift do modelo.
    Requer autenticação com API key com nível 'admin'.
    """
    # Verificar se o usuário tem permissão de admin
    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permissão insuficiente para acessar análise de drift. Requer API key com nível 'admin'."
        )
    
    # Verificar se o monitoramento está disponível
    if not MONITORING_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Monitoramento de drift não está disponível. Verifique a instalação dos módulos de monitoramento."
        )
    
    try:
        if refresh:
            # Gerar novo relatório de drift
            drift_report = generate_drift_report()
        else:
            # Obter relatório mais recente
            drift_report = get_latest_drift_report()
            
        return drift_report
    
    except Exception as e:
        logger.exception(f"Erro na análise de drift: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao analisar drift do modelo: {str(e)}"
        )

@router.get(
    "/drift/visualization",
    status_code=status.HTTP_200_OK,
    description="Retorna visualização de drift para uma feature específica"
)
async def visualize_drift(
    request: Request,
    feature: str = Query(..., description="Nome da feature para visualizar"),
    role: str = Depends(verify_api_key)
):
    """
    Endpoint para obter visualização de drift para uma feature específica.
    Retorna uma imagem PNG com a visualização.
    Requer autenticação com API key com nível 'admin'.
    """
    # Verificar se o usuário tem permissão de admin
    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permissão insuficiente para acessar visualização de drift. Requer API key com nível 'admin'."
        )
    
    # Verificar se o monitoramento está disponível
    if not MONITORING_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Monitoramento de drift não está disponível. Verifique a instalação dos módulos de monitoramento."
        )
    
    try:
        # Gerar visualização
        fig = visualize_feature_drift(feature)
        
        # Salvar a figura em um arquivo temporário
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        image_path = TMP_DIR / f"drift_{feature}_{timestamp}.png"
        fig.savefig(image_path, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        
        # Retornar a imagem
        return FileResponse(
            path=str(image_path),
            media_type="image/png",
            filename=f"drift_{feature}.png"
        )
    
    except Exception as e:
        logger.exception(f"Erro na visualização de drift: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao visualizar drift da feature {feature}: {str(e)}"
        )

@router.get(
    "/metrics/history",
    status_code=status.HTTP_200_OK,
    description="Retorna o histórico de métricas do modelo"
)
async def get_model_metrics_history(
    request: Request, 
    role: str = Depends(verify_api_key),
    days: int = Query(30, description="Número de dias para filtrar o histórico")
):
    """
    Endpoint para obter histórico de métricas do modelo.
    Requer autenticação com API key com nível 'admin'.
    """
    # Verificar se o usuário tem permissão de admin
    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permissão insuficiente para acessar histórico de métricas. Requer API key com nível 'admin'."
        )
    
    # Verificar se o monitoramento está disponível
    if not MONITORING_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Monitoramento de métricas não está disponível. Verifique a instalação dos módulos de monitoramento."
        )
    
    try:
        # Obter histórico de métricas
        metrics_data = get_metrics_history()
        
        # Filtrar por data, se necessário
        if days > 0:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
            cutoff_str = cutoff_date.isoformat()
            
            # Filtrar apenas métricas mais recentes que o limite
            metrics_history = [
                entry for entry in metrics_data.get("metrics_history", [])
                if entry.get("timestamp", "0") >= cutoff_str
            ]
            
            metrics_data["metrics_history"] = metrics_history
        
        return metrics_data
    
    except Exception as e:
        logger.exception(f"Erro ao obter histórico de métricas: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter histórico de métricas: {str(e)}"
        )

@router.get(
    "/predictions/recent",
    status_code=status.HTTP_200_OK,
    description="Retorna estatísticas sobre predições recentes"
)
async def get_recent_predictions_stats(
    request: Request, 
    role: str = Depends(verify_api_key),
    days: int = Query(7, description="Número de dias para filtrar predições")
):
    """
    Endpoint para obter estatísticas sobre predições recentes.
    Requer autenticação com API key com nível 'admin'.
    """
    # Verificar se o usuário tem permissão de admin
    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permissão insuficiente para acessar estatísticas de predições. Requer API key com nível 'admin'."
        )
    
    # Verificar se o monitoramento está disponível
    if not MONITORING_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Monitoramento de predições não está disponível. Verifique a instalação dos módulos de monitoramento."
        )
    
    try:
        # Obter predições recentes
        df = get_recent_predictions(days=days)
        
        if len(df) == 0:
            return {
                "count": 0,
                "message": "Não há predições registradas no período especificado."
            }
        
        # Calcular estatísticas
        stats = {
            "count": len(df),
            "period": {
                "start": df['timestamp'].min(),
                "end": df['timestamp'].max(),
            },
            "predictions": {
                "positive_rate": (df['prediction'] == 1).mean() * 100,
                "negative_rate": (df['prediction'] == 0).mean() * 100,
                "average_probability": df['prediction_probability'].mean()
            },
            "segments": df['segment'].value_counts().to_dict()
        }
        
        return stats
    
    except Exception as e:
        logger.exception(f"Erro ao obter estatísticas de predições: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter estatísticas de predições: {str(e)}"
        )

@router.post(
    "/run-pipeline",
    status_code=status.HTTP_202_ACCEPTED,
    description="Executa o pipeline de treinamento do modelo"
)
async def run_pipeline(
    request: Request, 
    background_tasks: BackgroundTasks,
    role: str = Depends(verify_api_key)
):
    """
    Endpoint para executar o pipeline de treinamento do modelo.
    Requer autenticação com API key com nível 'admin'.
    """
    # Verificar se o usuário tem permissão de admin
    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permissão insuficiente para executar o pipeline. Requer API key com nível 'admin'."
        )
    
    try:
        # Variável global para manter o estado do pipeline
        if not hasattr(run_pipeline, "running"):
            run_pipeline.running = False
            run_pipeline.start_time = None
            run_pipeline.end_time = None
            run_pipeline.success = None
            run_pipeline.output = None
        
        # Verificar se o pipeline já está em execução
        if run_pipeline.running:
            return {
                "status": "in_progress",
                "message": "Pipeline de treinamento já está em execução.",
                "start_time": run_pipeline.start_time.isoformat() if run_pipeline.start_time else None
            }
        
        # Executar o script run_pipeline.sh em background usando BackgroundTasks
        def run_script():
            logger.info("Iniciando execução do pipeline de treinamento...")
            run_pipeline.running = True
            run_pipeline.start_time = datetime.datetime.now()
            
            try:
                # Executa com timeout maior para evitar problemas
                result = subprocess.run(
                    ["./scripts/run_pipeline.sh"],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minutos de timeout
                )
                run_pipeline.output = {
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                run_pipeline.success = (result.returncode == 0)
                
                if result.returncode == 0:
                    logger.info("Pipeline de treinamento concluído com sucesso.")
                else:
                    logger.error(f"Erro ao executar pipeline: {result.stderr}")
            except subprocess.TimeoutExpired:
                logger.error("Timeout ao executar pipeline de treinamento")
                run_pipeline.success = False
                run_pipeline.output = {"error": "Timeout ao executar pipeline"}
            except Exception as e:
                logger.exception(f"Erro na execução do pipeline: {str(e)}")
                run_pipeline.success = False
                run_pipeline.output = {"error": str(e)}
            
            run_pipeline.running = False
            run_pipeline.end_time = datetime.datetime.now()
        
        # Usar BackgroundTasks para não bloquear a API
        background_tasks.add_task(run_script)
        
        return {
            "status": "accepted",
            "message": "Pipeline de treinamento iniciado em background. Este processo pode levar alguns minutos.",
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.exception(f"Erro ao iniciar pipeline de treinamento: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao iniciar pipeline de treinamento: {str(e)}"
        )

@router.get(
    "/pipeline-status",
    status_code=status.HTTP_200_OK,
    description="Verifica o status do pipeline de treinamento"
)
async def get_pipeline_status(
    request: Request,
    role: str = Depends(verify_api_key)
):
    """
    Endpoint para verificar o status do pipeline de treinamento.
    """
    # Verificar se o pipeline foi inicializado
    if not hasattr(run_pipeline, "running"):
        return {
            "status": "not_started",
            "message": "Pipeline não foi iniciado ainda."
        }
    
    # Retornar o status atual
    status_response = {
        "running": run_pipeline.running,
        "start_time": run_pipeline.start_time.isoformat() if run_pipeline.start_time else None,
        "end_time": run_pipeline.end_time.isoformat() if run_pipeline.end_time else None,
    }
    
    if not run_pipeline.running and run_pipeline.end_time:
        status_response["success"] = run_pipeline.success
        
        if role == "admin" and run_pipeline.output:
            status_response["output_summary"] = {
                "stdout_lines": len(run_pipeline.output.get("stdout", "").split("\n")),
                "stderr_lines": len(run_pipeline.output.get("stderr", "").split("\n")),
            }
    
    return status_response
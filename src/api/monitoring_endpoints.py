"""
monitoring_endpoints.py - Endpoints adicionais para monitoramento do modelo

Este módulo contém endpoints adicionais para monitorar o desempenho e drift do modelo.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, status, Request
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
import os
import time
import datetime
import logging
import uuid
import io
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
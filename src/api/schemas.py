"""Definição dos esquemas de dados para a API"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import datetime

class EducationLevel(str, Enum):
    """Níveis de educação suportados"""
    FUNDAMENTAL = "ensino_fundamental"
    MEDIO = "ensino_medio"
    SUPERIOR = "ensino_superior"
    POS = "pos_graduacao"


class CandidateRequest(BaseModel):
    """Esquema para solicitação de predição para um candidato"""
    
    # Campos obrigatórios
    idade: int = Field(..., ge=18, le=100, description="Idade do candidato")
    experiencia: float = Field(..., ge=0, description="Anos de experiência profissional")
    
    # Campos opcionais
    educacao: Optional[str] = Field(None, description="Nível de educação")
    tempo_desempregado: Optional[float] = Field(None, ge=0, description="Tempo desempregado em anos")
    area_formacao: Optional[str] = Field(None, description="Área de formação")
    habilidades: Optional[List[str]] = Field(None, description="Lista de habilidades do candidato")
    cargo_anterior: Optional[str] = Field(None, description="Cargo mais recente")
    salario_anterior: Optional[float] = Field(None, ge=0, description="Salário anterior em reais")
    anos_estudo: Optional[int] = Field(None, ge=0, le=30, description="Anos de estudo formal")
    
    # Campo para a vaga
    vaga_id: Optional[str] = Field(None, description="ID da vaga a qual o candidato está concorrendo")
    vaga_titulo: Optional[str] = Field(None, description="Título da vaga")
    vaga_area: Optional[str] = Field(None, description="Área da vaga (ex: TI, Vendas, Marketing)")
    vaga_senioridade: Optional[str] = Field(None, description="Nível de senioridade requerido")
    
    # Campo para extensão com outras propriedades
    extra_data: Optional[Dict[str, Any]] = Field(None, description="Dados adicionais do candidato")
    
    @validator('educacao')
    def validate_education(cls, v):
        """Valida e normaliza o nível educacional"""
        if not v:
            return None
            
        v = str(v).lower()
        
        # Mapeamento para valores padrão
        education_mapping = {
            'fundamental': 'ensino_fundamental',
            'ensino fundamental': 'ensino_fundamental',
            'medio': 'ensino_medio', 
            'ensino médio': 'ensino_medio',
            'ensino medio': 'ensino_medio',
            'superior': 'ensino_superior',
            'ensino superior': 'ensino_superior',
            'graduacao': 'ensino_superior',
            'graduação': 'ensino_superior',
            'pos': 'pos_graduacao',
            'pós': 'pos_graduacao',
            'pos graduacao': 'pos_graduacao',
            'pós graduação': 'pos_graduacao',
            'pos-graduacao': 'pos_graduacao',
            'mestrado': 'pos_graduacao',
            'doutorado': 'pos_graduacao'
        }
        
        normalized = education_mapping.get(v, v)
        
        # Verificar se o valor normalizado está entre os aceitos
        valid_values = [e.value for e in EducationLevel]
        if normalized not in valid_values:
            valid_values_str = ", ".join(valid_values)
            raise ValueError(
                f"Valor de educação '{v}' inválido. Use um dos seguintes: {valid_values_str}"
            )
            
        return normalized


class BatchCandidateRequest(BaseModel):
    """Esquema para solicitação de predição em lote"""
    candidates: List[CandidateRequest] = Field(
        ..., 
        min_items=1, 
        max_items=100,
        description="Lista de candidatos para predição em lote"
    )


class PredictionResponse(BaseModel):
    """Esquema para resposta de predição"""
    prediction: int = Field(..., description="Predição do modelo (0: não recomendado, 1: recomendado)")
    probability: float = Field(..., ge=0, le=1, description="Probabilidade da classe positiva")
    recommendation: str = Field(..., description="Recomendação textual baseada na predição")
    vaga_info: Optional[Dict[str, Any]] = Field(None, description="Informações sobre a vaga utilizada para predição")
    match_score: Optional[float] = Field(None, ge=0, le=1, description="Pontuação de compatibilidade candidato-vaga")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.85,
                "recommendation": "Recomendado",
                "vaga_info": {
                    "id": "vaga-123",
                    "titulo": "Desenvolvedor Python",
                    "area": "tecnologia",
                    "senioridade": "pleno"
                },
                "match_score": 0.78
            }
        }


class BatchPredictionResponse(BaseModel):
    """Esquema para resposta de predição em lote"""
    results: List[PredictionResponse] = Field(..., description="Lista de resultados de predição")
    timestamp: str = Field(..., description="Timestamp da predição")
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "prediction": 1,
                        "probability": 0.85,
                        "recommendation": "Recomendado"
                    },
                    {
                        "prediction": 0,
                        "probability": 0.35,
                        "recommendation": "Não recomendado"
                    }
                ],
                "timestamp": "2023-06-01T12:34:56Z"
            }
        }


class HealthResponse(BaseModel):
    """Esquema para resposta de verificação de saúde"""
    status: str = Field(..., description="Status da API")
    version: str = Field(..., description="Versão da API")
    model_info: Dict[str, Any] = Field(..., description="Informações sobre o modelo")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_info": {
                    "model_type": "RandomForestClassifier",
                    "n_features": 42,
                    "last_trained": "2023-05-15"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Esquema para resposta de erro"""
    error: str = Field(..., description="Mensagem de erro")
    detail: Optional[str] = Field(None, description="Detalhes adicionais do erro")
    timestamp: str = Field(..., description="Timestamp do erro")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Dados inválidos",
                "detail": "O campo 'idade' deve ser um número inteiro positivo",
                "timestamp": "2023-06-01T12:34:56Z"
            }
        }


class MetricsResponse(BaseModel):
    """Esquema para resposta de métricas da API"""
    uptime: str = Field(..., description="Tempo de atividade da API")
    request_count: int = Field(..., description="Número total de requisições")
    average_latency: float = Field(..., description="Latência média em milissegundos")
    error_rate: float = Field(..., description="Taxa de erro (%)") 
    model_metrics: Dict[str, float] = Field(..., description="Métricas do modelo em produção")
    
    class Config:
        schema_extra = {
            "example": {
                "uptime": "5d 12h 34m",
                "request_count": 15432,
                "average_latency": 127.45,
                "error_rate": 0.42,
                "model_metrics": {
                    "accuracy": 0.82,
                    "precision": 0.75,
                    "recall": 0.81,
                    "f1_score": 0.78
                }
            }
        }
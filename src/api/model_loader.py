"""Utilitário para carregar o modelo treinado"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any

# Caminho para o modelo treinado
MODEL_PATH = 'models/scoring_model.pkl'
SCALER_PATH = 'models/feature_scaler.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'

# Variáveis para caching
_model = None
_scaler = None
_encoder = None


def load_model():
    """
    Carrega o modelo de scoring, usando cache se já foi carregado anteriormente
    
    Returns:
        O modelo treinado
    """
    global _model
    
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}. Execute o treinamento primeiro.")
        
        print(f"Carregando modelo de {MODEL_PATH}...")
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
    
    return _model


def load_scaler():
    """
    Carrega o scaler para normalização de features, se existir
    
    Returns:
        O scaler para normalização ou None se não existir
    """
    global _scaler
    
    if _scaler is None and os.path.exists(SCALER_PATH):
        print(f"Carregando scaler de {SCALER_PATH}...")
        with open(SCALER_PATH, 'rb') as f:
            _scaler = pickle.load(f)
    
    return _scaler


def load_encoder():
    """
    Carrega o encoder para codificação de labels, se existir
    
    Returns:
        O encoder para labels ou None se não existir
    """
    global _encoder
    
    if _encoder is None and os.path.exists(ENCODER_PATH):
        print(f"Carregando encoder de {ENCODER_PATH}...")
        with open(ENCODER_PATH, 'rb') as f:
            _encoder = pickle.load(f)
    
    return _encoder


def get_feature_list() -> List[str]:
    """
    Obtém a lista de features utilizadas pelo modelo
    
    Returns:
        Lista de nomes de features
    """
    model = load_model()
    
    # Tenta obter as features do modelo
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier', None), 'feature_names_in_'):
        return list(model.named_steps['classifier'].feature_names_in_)
    else:
        # Se não for possível obter do modelo, retorna uma lista default
        # baseada na estrutura do projeto
        print("⚠️ Não foi possível determinar a lista exata de features do modelo.")
        print("⚠️ Utilizando lista padrão baseada na estrutura do projeto.")
        
        # Lista baseada nos dados processados
        try:
            df = pd.read_csv('data/processed/complete_processed_data.csv', nrows=1)
            return [col for col in df.columns if col != 'target']
        except:
            # Se não encontrar os dados, retorna uma lista genérica
            return ["idade", "experiencia", "educacao", "habilidades"]


def predict(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Realiza a predição para novos dados
    
    Args:
        data: DataFrame com os dados de entrada
        
    Returns:
        Dicionário com os resultados da predição:
        - prediction: 0 ou 1 (classe prevista)
        - probability: probabilidade da classe positiva
        - recommendation: texto com a recomendação baseada na predição
    """
    model = load_model()
    
    # Realizar a predição
    try:
        prediction = int(model.predict(data)[0])
        
        # Obter a probabilidade (assumindo modelo com predict_proba)
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(data)[:, 1][0])
        else:
            # Se o modelo não tiver predict_proba, usar predict como probabilidade
            probability = float(prediction)
        
        # Determinar recomendação baseada na predição
        if prediction == 1:
            recommendation = "Recomendado"
        else:
            recommendation = "Não recomendado"
        
        return {
            "prediction": prediction,
            "probability": probability,
            "recommendation": recommendation
        }
    except Exception as e:
        raise RuntimeError(f"Erro ao fazer predição: {str(e)}")
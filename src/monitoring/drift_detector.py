"""
drift_detector.py - Sistema de detecção de drift do modelo

Este módulo fornece funções para detectar mudanças na distribuição de dados
e no desempenho do modelo ao longo do tempo. Ele compara as distribuições
de dados de treinamento com as distribuições em produção para identificar
mudanças significativas que podem afetar o desempenho do modelo.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from scipy import stats

# Import local modules
from src.monitoring.metrics_store import get_recent_predictions, METRICS_DIR

# Diretório para armazenar os relatórios de drift
DRIFT_REPORTS_DIR = METRICS_DIR / "drift_reports"
TRAINING_STATS_FILE = METRICS_DIR / "training_statistics.json"

def initialize_drift_detector():
    """Inicializa o detector de drift, criando diretórios e arquivos necessários."""
    os.makedirs(DRIFT_REPORTS_DIR, exist_ok=True)

def calculate_feature_statistics(df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calcula estatísticas para cada feature no DataFrame.
    
    Args:
        df: DataFrame contendo os dados
        feature_names: Lista de nomes das features para analisar
        
    Returns:
        Dicionário com estatísticas para cada feature
    """
    stats_dict = {}
    
    for feature in feature_names:
        if feature in df.columns:
            feature_data = df[feature].dropna()
            
            # Skip empty features
            if len(feature_data) == 0:
                continue
                
            # Check if the feature is numeric
            if pd.api.types.is_numeric_dtype(feature_data):
                stats_dict[feature] = {
                    'mean': float(feature_data.mean()),
                    'median': float(feature_data.median()),
                    'std': float(feature_data.std() if len(feature_data) > 1 else 0),
                    'min': float(feature_data.min()),
                    'max': float(feature_data.max()),
                    'q1': float(feature_data.quantile(0.25)),
                    'q3': float(feature_data.quantile(0.75)),
                    'type': 'numeric'
                }
            else:
                # For categorical features
                value_counts = feature_data.value_counts(normalize=True).to_dict()
                stats_dict[feature] = {
                    'value_distribution': {str(k): float(v) for k, v in value_counts.items()},
                    'n_unique': len(value_counts),
                    'most_common': str(feature_data.value_counts().index[0]),
                    'type': 'categorical'
                }
    
    return stats_dict

def save_training_statistics(train_df: pd.DataFrame, feature_names: List[str]):
    """
    Salva estatísticas do conjunto de treinamento para referência futura.
    
    Args:
        train_df: DataFrame com os dados de treinamento
        feature_names: Lista de features relevantes para o modelo
    """
    stats = calculate_feature_statistics(train_df, feature_names)
    
    # Adicionar metadados
    stats_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'n_samples': len(train_df),
        'feature_statistics': stats
    }
    
    # Salvar estatísticas
    with open(TRAINING_STATS_FILE, 'w') as f:
        json.dump(stats_data, f, indent=2)

def load_training_statistics() -> Dict[str, Any]:
    """
    Carrega as estatísticas do conjunto de treinamento.
    
    Returns:
        Dicionário com estatísticas do conjunto de treinamento
    """
    try:
        with open(TRAINING_STATS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            'timestamp': None,
            'n_samples': 0,
            'feature_statistics': {}
        }

def detect_distribution_drift(
    recent_data: pd.DataFrame, 
    feature_names: Optional[List[str]] = None, 
    threshold: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    """
    Detecta drift na distribuição das features entre dados de treinamento e produção.
    
    Args:
        recent_data: DataFrame com dados recentes de produção
        feature_names: Lista de features para analisar (se None, usa todas)
        threshold: Limiar para considerar uma mudança significativa
        
    Returns:
        Dicionário com resultados da detecção de drift para cada feature
    """
    # Carregar estatísticas de treinamento
    training_stats = load_training_statistics()
    training_feature_stats = training_stats.get('feature_statistics', {})
    
    if not training_feature_stats:
        return {'error': 'Estatísticas de treinamento não disponíveis'}
    
    # Se feature_names não for fornecido, usar todas as features disponíveis
    if feature_names is None:
        feature_names = list(training_feature_stats.keys())
    
    # Extrair features dos dados recentes
    if len(recent_data) == 0:
        return {'error': 'Não há dados recentes disponíveis para análise'}
        
    # Para dados armazenados com features como JSON
    if 'features_dict' in recent_data.columns:
        # Extrair features dos dicionários
        features_df = pd.DataFrame([])
        for feature in feature_names:
            if feature in training_feature_stats:
                try:
                    features_df[feature] = recent_data['features_dict'].apply(
                        lambda x: x.get(feature, np.nan) if isinstance(x, dict) else np.nan
                    )
                except:
                    continue
    else:
        # Usar colunas diretamente
        features_df = recent_data[feature_names].copy()
    
    drift_results = {}
    
    for feature in feature_names:
        if feature not in training_feature_stats or feature not in features_df.columns:
            continue
        
        feature_data = features_df[feature].dropna()
        
        # Skip if no data
        if len(feature_data) == 0:
            continue
            
        feature_type = training_feature_stats[feature].get('type', 'unknown')
        drift_detected = False
        drift_metrics = {}
        
        if feature_type == 'numeric':
            # For numeric features
            train_mean = training_feature_stats[feature].get('mean', 0)
            train_std = training_feature_stats[feature].get('std', 1)
            
            # Current statistics
            current_mean = feature_data.mean()
            current_std = feature_data.std() if len(feature_data) > 1 else 0
            
            # Standardized difference in means
            if train_std > 0:
                mean_diff_std = abs(train_mean - current_mean) / train_std
                drift_detected = mean_diff_std > 2.0  # More than 2 standard deviations
                
                drift_metrics = {
                    'train_mean': train_mean,
                    'current_mean': float(current_mean),
                    'train_std': train_std,
                    'current_std': float(current_std),
                    'standardized_mean_diff': float(mean_diff_std),
                    'drift_threshold': 2.0
                }
            
        elif feature_type == 'categorical':
            # For categorical features
            train_dist = training_feature_stats[feature].get('value_distribution', {})
            
            # Current distribution
            current_dist = feature_data.value_counts(normalize=True).to_dict()
            
            # JS divergence between distributions
            js_divergence = calculate_js_divergence(train_dist, current_dist)
            drift_detected = js_divergence > threshold
            
            drift_metrics = {
                'js_divergence': float(js_divergence),
                'drift_threshold': threshold,
                'n_categories_train': len(train_dist),
                'n_categories_current': len(current_dist),
                'new_categories': list(set(current_dist.keys()) - set(train_dist.keys()))
            }
        
        drift_results[feature] = {
            'drift_detected': drift_detected,
            'metrics': drift_metrics,
            'type': feature_type
        }
    
    return drift_results

def calculate_js_divergence(dist1: Dict[str, float], dist2: Dict[str, float]) -> float:
    """
    Calcula a divergência Jensen-Shannon entre duas distribuições de probabilidade.
    
    Args:
        dist1: Primeira distribuição (dicionário de valores para probabilidades)
        dist2: Segunda distribuição
        
    Returns:
        Valor da divergência JS entre 0 e 1
    """
    # Obter união de todas as chaves
    all_keys = set(dist1.keys()) | set(dist2.keys())
    
    # Normalizar distribuições para que somem 1
    def normalize_dist(dist):
        total = sum(dist.values())
        return {k: v / total if total > 0 else 0 for k, v in dist.items()}
    
    dist1 = normalize_dist(dist1)
    dist2 = normalize_dist(dist2)
    
    # Calcular as distribuições completas (preenchendo zeros)
    p = np.array([dist1.get(k, 0.0) for k in all_keys])
    q = np.array([dist2.get(k, 0.0) for k in all_keys])
    
    # Evitar zeros puros para o cálculo de KL
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    
    # Distribuição média
    m = (p + q) / 2
    
    # Calcular divergência JS
    js_div = (stats.entropy(p, m) + stats.entropy(q, m)) / 2
    
    return min(1.0, js_div)  # Limitar a 1.0 para normalização

def generate_drift_report() -> Dict[str, Any]:
    """
    Gera um relatório completo de drift do modelo e features.
    
    Returns:
        Dicionário com relatório completo
    """
    # Obter dados recentes
    recent_data = get_recent_predictions(days=30)
    
    # Carregar estatísticas de treinamento
    training_stats = load_training_statistics()
    feature_names = list(training_stats.get('feature_statistics', {}).keys())
    
    # Se não houver dados ou estatísticas, retornar relatório vazio
    if len(feature_names) == 0 or len(recent_data) == 0:
        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'status': 'error',
            'message': 'Dados insuficientes para análise de drift',
            'drift_detected': False
        }
    
    # Detectar drift nas distribuições das features
    feature_drift = detect_distribution_drift(recent_data, feature_names)
    
    # Analisar resultados para determinar se há drift
    features_with_drift = [f for f, res in feature_drift.items() 
                          if isinstance(res, dict) and res.get('drift_detected', False)]
    
    # Calcular score geral de drift (proporção de features com drift)
    drift_score = len(features_with_drift) / len(feature_names) if feature_names else 0
    
    # Criar relatório
    report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'drift_score': drift_score,
        'drift_detected': drift_score > 0.2,  # Se mais de 20% das features têm drift
        'n_samples_analyzed': len(recent_data),
        'features_analyzed': len(feature_names),
        'features_with_drift': features_with_drift,
        'feature_details': feature_drift
    }
    
    # Salvar relatório
    report_filename = f"drift_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(DRIFT_REPORTS_DIR / report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def get_latest_drift_report() -> Dict[str, Any]:
    """
    Recupera o relatório de drift mais recente, ou gera um novo se não houver.
    
    Returns:
        Dicionário com o relatório mais recente
    """
    try:
        # Listar todos os arquivos de relatório e pegar o mais recente
        reports = list(DRIFT_REPORTS_DIR.glob('drift_report_*.json'))
        
        if not reports:
            return generate_drift_report()
        
        latest_report = max(reports, key=lambda x: x.stat().st_mtime)
        
        with open(latest_report, 'r') as f:
            return json.load(f)
            
    except (FileNotFoundError, json.JSONDecodeError):
        return generate_drift_report()

def visualize_feature_drift(feature_name: str, save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualiza o drift de uma feature específica ao longo do tempo.
    
    Args:
        feature_name: Nome da feature para visualizar
        save_path: Caminho para salvar a figura (opcional)
        
    Returns:
        Objeto Matplotlib Figure
    """
    # Obter dados recentes
    recent_data = get_recent_predictions(days=90)  # Últimos 90 dias
    
    if len(recent_data) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Dados insuficientes para análise", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Carregar estatísticas de treinamento
    training_stats = load_training_statistics()
    training_feature_stats = training_stats.get('feature_statistics', {}).get(feature_name, {})
    
    if not training_feature_stats:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Estatísticas de treinamento não disponíveis para {feature_name}", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Extrair dados da feature
    if 'features_dict' in recent_data.columns:
        try:
            feature_data = recent_data['features_dict'].apply(
                lambda x: x.get(feature_name, np.nan) if isinstance(x, dict) else np.nan
            )
        except:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Não foi possível extrair dados para {feature_name}", 
                    ha='center', va='center', fontsize=14)
            return fig
    elif feature_name in recent_data.columns:
        feature_data = recent_data[feature_name]
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Feature {feature_name} não encontrada nos dados", 
                ha='center', va='center', fontsize=14)
        return fig
    
    feature_type = training_feature_stats.get('type', 'unknown')
    
    # Criar visualização
    fig, ax = plt.subplots(figsize=(12, 7))
    
    if feature_type == 'numeric':
        # Para features numéricas
        train_mean = training_feature_stats.get('mean', 0)
        train_std = training_feature_stats.get('std', 1)
        
        # Plotar distribuição dos dados recentes
        sns.histplot(feature_data.dropna(), kde=True, ax=ax)
        
        # Adicionar linhas para média e desvio padrão do treinamento
        ax.axvline(train_mean, color='red', linestyle='--', 
                   label=f'Média Treinamento: {train_mean:.2f}')
        ax.axvline(train_mean + train_std, color='green', linestyle=':', 
                   label=f'±1 Desvio Padrão: {train_std:.2f}')
        ax.axvline(train_mean - train_std, color='green', linestyle=':')
        
        ax.set_title(f'Distribuição de {feature_name} - Comparação com Treinamento', fontsize=14)
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Densidade')
        
    elif feature_type == 'categorical':
        # Para features categóricas
        train_dist = training_feature_stats.get('value_distribution', {})
        
        # Organizar dados para visualização
        current_dist = feature_data.value_counts(normalize=True).to_dict()
        
        # Obter todas as categorias (união)
        all_categories = list(set(train_dist.keys()) | set(current_dist.keys()))
        
        # Criar DataFrame para plotagem
        plot_data = pd.DataFrame({
            'Categoria': list(all_categories) * 2,
            'Distribuição': [train_dist.get(cat, 0) for cat in all_categories] + 
                           [current_dist.get(cat, 0) for cat in all_categories],
            'Fonte': ['Treinamento'] * len(all_categories) + 
                    ['Produção'] * len(all_categories)
        })
        
        # Plotar comparação de distribuições
        sns.barplot(x='Categoria', y='Distribuição', hue='Fonte', data=plot_data, ax=ax)
        
        ax.set_title(f'Distribuição de {feature_name} - Treinamento vs. Produção', fontsize=14)
        ax.set_xlabel('Categoria')
        ax.set_ylabel('Proporção')
        plt.xticks(rotation=45)
        
    else:
        ax.text(0.5, 0.5, f"Tipo de feature não suportado: {feature_type}", 
                ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    ax.legend()
    
    # Salvar figura se caminho fornecido
    if save_path:
        plt.savefig(save_path)
    
    return fig
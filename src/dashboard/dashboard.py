"""
dashboard.py - Dashboard para visualização de métricas e drift do modelo

Este script cria um dashboard interativo usando Streamlit para visualizar as métricas
e o drift do modelo de scoring da Decision.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
import datetime
import time
from pathlib import Path
import os
import sys

# Adicionar diretório raiz do projeto ao path para importar módulos locais
# Determinar o caminho do projeto raiz (dois diretórios acima do arquivo atual)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.monitoring.metrics_store import (
    get_metrics_history,
    get_recent_predictions,
    METRICS_FILE
)
from src.monitoring.drift_detector import (
    get_latest_drift_report,
    visualize_feature_drift,
    DRIFT_REPORTS_DIR
)

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Monitoramento - Decision Scoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar variáveis de sessão
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.datetime.now()

# Função para formatar timestamp
def format_timestamp(timestamp_str):
    try:
        dt = datetime.datetime.fromisoformat(timestamp_str)
        return dt.strftime("%d/%m/%Y %H:%M:%S")
    except:
        return timestamp_str

# Barra lateral com informações
with st.sidebar:
    st.title("Decision Scoring")
    st.subheader("Monitoramento do Modelo")
    
    st.markdown("---")
    
    # Botão de atualizar
    if st.button("🔄 Atualizar Dados"):
        st.session_state.last_refresh = datetime.datetime.now()
        st.success("Dados atualizados!")
    
    st.markdown(f"Última atualização: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    
    st.markdown("---")
    
    # Opções de navegação
    page = st.radio(
        "Navegação",
        ["Dashboard Principal", "Análise de Drift", "Histórico de Métricas", "Predições Recentes"]
    )
    
    st.markdown("---")
    
    # Informações do projeto
    st.markdown("**Informações do Projeto**")
    st.markdown("Sistema de Monitoramento para o modelo de scoring da Decision.")
    st.markdown("Versão: 1.0.0")

# Função para carregar métricas
@st.cache_data(ttl=60)  # Cache por 60 segundos
def load_metrics_data():
    try:
        metrics_data = get_metrics_history()
        return metrics_data
    except Exception as e:
        st.error(f"Erro ao carregar métricas: {str(e)}")
        return {"model_info": {}, "metrics_history": []}

# Função para carregar predições recentes
@st.cache_data(ttl=60)  # Cache por 60 segundos
def load_recent_predictions(days=30):
    try:
        predictions_df = get_recent_predictions(days=days)
        return predictions_df
    except Exception as e:
        st.error(f"Erro ao carregar predições: {str(e)}")
        return pd.DataFrame()

# Função para carregar relatório de drift
@st.cache_data(ttl=60)  # Cache por 60 segundos
def load_drift_report():
    try:
        drift_report = get_latest_drift_report()
        return drift_report
    except Exception as e:
        st.error(f"Erro ao carregar relatório de drift: {str(e)}")
        return {"timestamp": "", "drift_detected": False, "drift_score": 0.0}

# Dashboard Principal
def show_main_dashboard():
    st.title("Dashboard Principal")
    
    # Carregar dados
    metrics_data = load_metrics_data()
    predictions_df = load_recent_predictions(days=30)
    drift_report = load_drift_report()
    
    # Verificar se há dados suficientes
    if not metrics_data["metrics_history"]:
        st.warning("Não há dados de métricas disponíveis.")
        return
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    # Métricas mais recentes
    latest_metrics = metrics_data["metrics_history"][-1]["metrics"]
    
    with col1:
        st.metric(
            label="Acurácia do Modelo", 
            value=f"{latest_metrics.get('accuracy', 0.0) * 100:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Precisão", 
            value=f"{latest_metrics.get('precision', 0.0) * 100:.1f}%"
        )
    
    with col3:
        st.metric(
            label="Recall", 
            value=f"{latest_metrics.get('recall', 0.0) * 100:.1f}%"
        )
    
    with col4:
        st.metric(
            label="F1-Score", 
            value=f"{latest_metrics.get('f1_score', 0.0) * 100:.1f}%"
        )
    
    # Status do Drift
    drift_score = drift_report.get("drift_score", 0.0)
    drift_detected = drift_report.get("drift_detected", False)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Status do Drift")
        
        if drift_detected:
            st.error("⚠️ Drift Detectado!")
            st.markdown(f"**Score de Drift:** {drift_score * 100:.1f}%")
            st.markdown(f"**Features afetadas:** {len(drift_report.get('features_with_drift', []))}")
            st.markdown(f"**Última verificação:** {format_timestamp(drift_report.get('timestamp', ''))}")
        else:
            st.success("✅ Modelo Estável")
            st.markdown(f"**Score de Drift:** {drift_score * 100:.1f}%")
            st.markdown(f"**Última verificação:** {format_timestamp(drift_report.get('timestamp', ''))}")
    
    with col2:
        # Evolução das métricas
        if len(metrics_data["metrics_history"]) > 1:
            st.subheader("Evolução das Métricas")
            
            # Preparar dados para o gráfico
            dates = []
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            
            for entry in metrics_data["metrics_history"]:
                try:
                    dates.append(datetime.datetime.fromisoformat(entry["timestamp"]))
                    metrics = entry["metrics"]
                    accuracies.append(metrics.get("accuracy", None))
                    precisions.append(metrics.get("precision", None))
                    recalls.append(metrics.get("recall", None))
                    f1_scores.append(metrics.get("f1_score", None))
                except:
                    continue
            
            # Criar DataFrame para o gráfico
            metrics_df = pd.DataFrame({
                "Data": dates,
                "Acurácia": accuracies,
                "Precisão": precisions,
                "Recall": recalls,
                "F1-Score": f1_scores
            })
            
            # Plotar gráfico
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(metrics_df["Data"], metrics_df["Acurácia"], marker='o', label="Acurácia")
            ax.plot(metrics_df["Data"], metrics_df["Precisão"], marker='s', label="Precisão")
            ax.plot(metrics_df["Data"], metrics_df["Recall"], marker='^', label="Recall")
            ax.plot(metrics_df["Data"], metrics_df["F1-Score"], marker='*', label="F1-Score")
            
            ax.set_ylabel("Valor")
            ax.set_xlabel("Data")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Exibir gráfico
            st.pyplot(fig)
        else:
            st.info("Não há dados históricos suficientes para exibir evolução das métricas.")
    
    st.markdown("---")
    
    # Estatísticas de predições recentes
    st.subheader("Predições Recentes")
    
    if len(predictions_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição de predições (positivas vs negativas)
            fig, ax = plt.subplots(figsize=(8, 6))
            predictions_dist = predictions_df['prediction'].value_counts().to_dict()
            labels = ['Aprovado' if k == 1 else 'Reprovado' for k in predictions_dist.keys()]
            sizes = list(predictions_dist.values())
            
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
            
        with col2:
            # Estatísticas por segmento
            if 'segment' in predictions_df.columns:
                segment_stats = predictions_df.groupby('segment')['prediction'].agg(['mean', 'count'])
                segment_stats['mean'] = segment_stats['mean'] * 100  # Converter para percentual
                segment_stats.columns = ['Taxa de Aprovação (%)', 'Quantidade']
                segment_stats = segment_stats.sort_values('Quantidade', ascending=False)
                
                st.dataframe(segment_stats.style.format({'Taxa de Aprovação (%)': '{:.1f}%'}))
            else:
                st.info("Não há dados de segmentos disponíveis.")
    else:
        st.info("Não há dados de predições recentes disponíveis.")

# Análise de Drift
def show_drift_analysis():
    st.title("Análise de Drift do Modelo")
    
    # Carregar relatório de drift
    drift_report = load_drift_report()
    
    if not drift_report:
        st.warning("Não há relatório de drift disponível.")
        return
    
    # Informações gerais
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Score de Drift", 
            value=f"{drift_report.get('drift_score', 0.0) * 100:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Status", 
            value="Drift Detectado!" if drift_report.get("drift_detected", False) else "Estável"
        )
    
    with col3:
        st.metric(
            label="Features com Drift", 
            value=len(drift_report.get("features_with_drift", []))
        )
    
    st.markdown("---")
    
    # Detalhes do drift
    if drift_report.get("features_with_drift", []):
        st.subheader("Features com Drift Detectado")
        
        # Lista de features com drift
        for feature in drift_report.get("features_with_drift", []):
            feature_details = drift_report.get("feature_details", {}).get(feature, {})
            
            if feature_details:
                with st.expander(f"Feature: {feature}"):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.markdown(f"**Tipo:** {feature_details.get('type', 'N/A')}")
                        
                        # Métricas específicas por tipo
                        metrics = feature_details.get("metrics", {})
                        if feature_details.get('type') == 'numeric':
                            st.markdown(f"**Média (Treino):** {metrics.get('train_mean', 0.0):.2f}")
                            st.markdown(f"**Média (Atual):** {metrics.get('current_mean', 0.0):.2f}")
                            st.markdown(f"**Diferença Padronizada:** {metrics.get('standardized_mean_diff', 0.0):.2f}")
                        elif feature_details.get('type') == 'categorical':
                            st.markdown(f"**Divergência JS:** {metrics.get('js_divergence', 0.0):.3f}")
                            st.markdown(f"**Categorias (Treino):** {metrics.get('n_categories_train', 0)}")
                            st.markdown(f"**Categorias (Atual):** {metrics.get('n_categories_current', 0)}")
                            
                            if metrics.get('new_categories', []):
                                st.markdown("**Novas Categorias:**")
                                for cat in metrics.get('new_categories', []):
                                    st.markdown(f"- {cat}")
                    
                    with col2:
                        # Visualizar drift da feature
                        try:
                            fig = visualize_feature_drift(feature)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Erro ao visualizar drift: {str(e)}")
    else:
        st.success("Não há drift significativo detectado nas features.")
    
    # Informações sobre a análise
    st.markdown("---")
    st.subheader("Informações da Análise")
    st.markdown(f"**Data da análise:** {format_timestamp(drift_report.get('timestamp', ''))}")
    st.markdown(f"**Amostras analisadas:** {drift_report.get('n_samples_analyzed', 0)}")
    st.markdown(f"**Features analisadas:** {drift_report.get('features_analyzed', 0)}")

# Histórico de Métricas
def show_metrics_history():
    st.title("Histórico de Métricas do Modelo")
    
    # Carregar dados
    metrics_data = load_metrics_data()
    
    if not metrics_data["metrics_history"]:
        st.warning("Não há dados de métricas disponíveis.")
        return
    
    # Informações do modelo
    st.subheader("Informações do Modelo")
    model_info = metrics_data.get("model_info", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Data de criação:** {format_timestamp(model_info.get('creation_date', ''))}")
        st.markdown(f"**Versão do modelo:** {model_info.get('model_version', 'N/A')}")
    
    with col2:
        baseline_metrics = model_info.get('baseline_metrics', {})
        if baseline_metrics:
            st.markdown("**Métricas Baseline:**")
            for metric, value in baseline_metrics.items():
                st.markdown(f"- {metric}: {value:.4f}")
    
    st.markdown("---")
    
    # Gráfico de evolução das métricas
    st.subheader("Evolução das Métricas")
    
    # Preparar dados para o gráfico
    metrics_history = metrics_data["metrics_history"]
    
    if len(metrics_history) > 1:
        # Criar DataFrame para análise
        metrics_rows = []
        
        for entry in metrics_history:
            try:
                timestamp = datetime.datetime.fromisoformat(entry["timestamp"])
                metrics = entry["metrics"]
                
                row = {"timestamp": timestamp}
                for metric_name, metric_value in metrics.items():
                    row[metric_name] = metric_value
                
                metrics_rows.append(row)
            except:
                continue
        
        metrics_df = pd.DataFrame(metrics_rows)
        
        # Seleção de métricas para exibir
        available_metrics = [col for col in metrics_df.columns if col != 'timestamp']
        selected_metrics = st.multiselect(
            "Selecione as métricas para visualizar",
            available_metrics,
            default=['accuracy', 'precision', 'recall', 'f1_score'][:min(len(available_metrics), 4)]
        )
        
        if selected_metrics:
            # Plotar gráfico
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for metric in selected_metrics:
                ax.plot(metrics_df['timestamp'], metrics_df[metric], marker='o', label=metric)
            
            ax.set_ylabel("Valor")
            ax.set_xlabel("Data")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Formatar eixo X para mostrar datas
            plt.xticks(rotation=45)
            
            # Exibir gráfico
            st.pyplot(fig)
            
            # Exibir tabela com dados
            st.subheader("Dados Históricos")
            
            display_df = metrics_df[['timestamp'] + selected_metrics].copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime("%Y-%m-%d %H:%M")
            
            st.dataframe(display_df.sort_values('timestamp', ascending=False))
        else:
            st.info("Selecione pelo menos uma métrica para visualizar o gráfico.")
    else:
        st.info("Não há dados históricos suficientes para exibir evolução das métricas.")

# Predições Recentes
def show_recent_predictions():
    st.title("Análise de Predições Recentes")
    
    # Seletor de período
    days = st.slider("Selecione o período de análise (dias)", 1, 90, 30)
    
    # Carregar dados
    predictions_df = load_recent_predictions(days=days)
    
    if len(predictions_df) == 0:
        st.warning("Não há dados de predições disponíveis para o período selecionado.")
        return
    
    # Métricas gerais
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total de Predições", 
            value=len(predictions_df)
        )
    
    with col2:
        approval_rate = (predictions_df['prediction'] == 1).mean() * 100
        st.metric(
            label="Taxa de Aprovação", 
            value=f"{approval_rate:.1f}%"
        )
    
    with col3:
        avg_probability = predictions_df['prediction_probability'].mean() * 100
        st.metric(
            label="Probabilidade Média", 
            value=f"{avg_probability:.1f}%"
        )
    
    st.markdown("---")
    
    # Análise por segmentos
    if 'segment' in predictions_df.columns:
        st.subheader("Análise por Segmento")
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Tabela de estatísticas por segmento
            segment_stats = predictions_df.groupby('segment').agg({
                'prediction': ['count', 'mean'],
                'prediction_probability': 'mean'
            })
            
            segment_stats.columns = ['Quantidade', 'Taxa Aprovação', 'Prob. Média']
            segment_stats['Taxa Aprovação'] = segment_stats['Taxa Aprovação'] * 100
            segment_stats['Prob. Média'] = segment_stats['Prob. Média'] * 100
            segment_stats = segment_stats.sort_values('Quantidade', ascending=False)
            
            st.dataframe(segment_stats.style.format({
                'Taxa Aprovação': '{:.1f}%',
                'Prob. Média': '{:.1f}%'
            }))
        
        with col2:
            # Gráfico de barras para comparação
            fig, ax = plt.subplots(figsize=(10, 6))
            
            segment_plot_df = segment_stats.reset_index()
            sns.barplot(x='segment', y='Taxa Aprovação', data=segment_plot_df, ax=ax)
            
            ax.set_title('Taxa de Aprovação por Segmento')
            ax.set_ylabel('Taxa de Aprovação (%)')
            ax.set_xlabel('Segmento')
            
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
    
    # Análise temporal
    st.subheader("Análise Temporal")
    
    # Converter timestamp para datetime
    predictions_df['date'] = pd.to_datetime(predictions_df['timestamp']).dt.date
    
    # Agrupar por data
    daily_stats = predictions_df.groupby('date').agg({
        'prediction': ['count', 'mean'],
        'prediction_probability': 'mean'
    })
    
    daily_stats.columns = ['Quantidade', 'Taxa Aprovação', 'Prob. Média']
    daily_stats['Taxa Aprovação'] = daily_stats['Taxa Aprovação'] * 100
    daily_stats['Prob. Média'] = daily_stats['Prob. Média'] * 100
    
    # Plotar gráfico
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Eixo para quantidade
    color = 'tab:blue'
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Quantidade de Predições', color=color)
    ax1.bar(daily_stats.index, daily_stats['Quantidade'], color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Eixo secundário para taxa de aprovação
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Taxa de Aprovação (%)', color=color)
    ax2.plot(daily_stats.index, daily_stats['Taxa Aprovação'], color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Evolução diária de predições e aprovações')
    plt.xticks(rotation=45)
    
    st.pyplot(fig)
    
    # Tabela com dados diários
    st.subheader("Dados Diários")
    st.dataframe(daily_stats.style.format({
        'Taxa Aprovação': '{:.1f}%',
        'Prob. Média': '{:.1f}%'
    }))

# Exibir a página selecionada
if page == "Dashboard Principal":
    show_main_dashboard()
elif page == "Análise de Drift":
    show_drift_analysis()
elif page == "Histórico de Métricas":
    show_metrics_history()
elif page == "Predições Recentes":
    show_recent_predictions()

# Footer
st.markdown("---")
st.markdown("Decision Scoring Dashboard © 2025")
st.markdown("Desenvolvido para monitoramento contínuo de performance e drift do modelo.")
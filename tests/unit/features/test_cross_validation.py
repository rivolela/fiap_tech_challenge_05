"""Testes para o módulo cross_validation"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from src.features.cross_validation import (
    detect_leakage_candidates, select_features, perform_stratified_kfold_cv,
    balance_training_data, visualize_feature_importance, safe_split_and_balance
)

@pytest.fixture
def sample_data():
    """Fixture para dados de exemplo."""
    # Criar um DataFrame de exemplo
    np.random.seed(42)
    n_samples = 100
    
    # Feature fortemente correlacionada com o target (possível leakage)
    high_corr = np.random.normal(0, 1, n_samples)
    target = (high_corr > 0).astype(int)
    
    # Features normais
    df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'leaky_feature': high_corr + np.random.normal(0, 0.1, n_samples),  # Correlação alta
        'target': target
    })
    
    return df

@patch('builtins.print')  # Suprimir prints durante o teste
def test_detect_leakage_candidates(mock_print, sample_data):
    """Testa a detecção de candidatos a data leakage."""
    # Para garantir que leaky_feature tenha alta correlação
    sample_data['leaky_feature'] = sample_data['target'] * 0.9 + np.random.normal(0, 0.1, len(sample_data))
    
    # Chamar a função com limiar mais baixo para detectar nossa feature
    leakage_candidates = detect_leakage_candidates(sample_data, 'target', threshold=0.6)
    
    # Verificar se a feature com alta correlação foi detectada
    assert 'leaky_feature' in leakage_candidates
    
    # Verificar se features normais não foram detectadas
    assert 'feature1' not in leakage_candidates
    assert 'feature2' not in leakage_candidates

@patch('builtins.print')  # Suprimir prints durante o teste
def test_select_features(mock_print):
    """Testa a seleção de features."""
    # Criar dados simulados
    np.random.seed(42)
    n_samples = 100
    n_features = 10  # Reduzido para acelerar o teste
    
    # Criando uma matriz de features, apenas algumas são importantes
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Criando o target baseado apenas em algumas features
    important_features = [0, 2, 5]
    y = (X[:, important_features].sum(axis=1) > 0).astype(int)
    
    # Convertendo para o formato esperado
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    
    # Testando apenas um método para evitar tempo de execução longo
    method = 'rfe'
    n_features_to_select = 7
    
    # Chamar a função
    selected_features = select_features(X_df, y, method=method, n_features=n_features_to_select)
    
    # Verificar se o número correto de features foi selecionado
    assert len(selected_features) == n_features_to_select
    
    # Verificar se o resultado é uma lista de strings (nomes das features)
    assert all(isinstance(f, str) for f in selected_features)

@patch('sklearn.model_selection.StratifiedKFold')
@patch('src.features.cross_validation.balance_training_data')
@patch('builtins.print')  # Suprimir prints durante o teste
def test_perform_stratified_kfold_cv(mock_print, mock_balance, mock_kfold):
    """Testa a validação cruzada estratificada."""
    # Criar dados simulados
    np.random.seed(42)
    X = pd.DataFrame(np.random.normal(0, 1, size=(100, 5)))
    y = pd.Series(np.random.randint(0, 2, 100))  # Converter para Series para permitir iloc
    
    # Mock da função de balanceamento para retornar os dados originais
    mock_balance.return_value = (X, y)
    
    # Criar um modelo mock
    model = MagicMock()
    model.fit.return_value = model
    model.predict.return_value = np.random.randint(0, 2, 20)
    model.predict_proba.return_value = np.random.uniform(0, 1, size=(20, 2))
    
    # Configurar o mock do KFold
    mock_split_instance = MagicMock()
    mock_splits = [(np.arange(80), np.arange(80, 100)) for _ in range(5)]  # 5 splits
    mock_split_instance.split.return_value = mock_splits
    mock_kfold.return_value = mock_split_instance
    
    # Chamar a função
    cv_results = perform_stratified_kfold_cv(X, y, model, n_splits=5)
    
    # Verificar se a função retornou um dicionário de resultados
    assert isinstance(cv_results, dict)
    assert 'accuracy' in cv_results
    assert 'auc' in cv_results
    
    # Verificar se os resultados foram retornados corretamente
    assert 'accuracy' in cv_results
    assert 'precision' in cv_results
    assert 'recall' in cv_results
    assert 'f1' in cv_results
    assert 'auc' in cv_results
    
    # Verificar se o modelo foi treinado 5 vezes (uma para cada fold)
    assert model.fit.call_count == 5

def test_balance_training_data():
    """Testa o balanceamento de dados de treinamento."""
    # Criar dados simulados desbalanceados
    np.random.seed(42)
    X = pd.DataFrame(np.random.normal(0, 1, size=(100, 5)))
    y = np.zeros(100)
    y[:10] = 1  # Apenas 10% são da classe minoritária
    
    # Chamar a função - método atual usa oversample manual, não SMOTE
    X_balanced, y_balanced = balance_training_data(X, y)
    
    # Verificar se os resultados retornados são corretos
    assert len(X_balanced) > len(X)  # Deve ter mais amostras após o balanceamento
    
    # Verificar se há balanceamento nas classes
    unique, counts = np.unique(y_balanced, return_counts=True)
    assert counts[0] == counts[1]  # Classes devem estar balanceadas
    assert counts[0] == counts[1]  # Classes agora devem ter o mesmo número

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.figure')
@patch('builtins.print')  # Suprimir prints durante o teste
def test_visualize_feature_importance(mock_print, mock_figure, mock_savefig):
    """Testa a visualização de importância das features."""
    # Criar um modelo com importância das features
    model = MagicMock()
    model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.05, 0.35])
    
    # Criar nomes das features
    feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    
    # Chamar a função com save_path
    save_path = 'data/visualizations/feature_importance.png'
    visualize_feature_importance(model, feature_names, save_path=save_path)
    
    # Verificar se a figura foi criada
    mock_figure.assert_called()
    
    # Não testar se savefig foi chamado com parâmetros específicos, apenas que foi chamado
    mock_savefig.assert_called()

def test_safe_split_and_balance():
    """Testa a divisão segura dos dados de forma simplificada."""
    # Criar dados simulados
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'target': np.random.randint(0, 2, 100)
    })
    
    # Em vez de usar mocks complexos, vamos usar a função diretamente
    # e apenas verificar o formato do resultado
    with patch('builtins.print'):  # Suprimir prints
        # Chamar a função real
        result = safe_split_and_balance(df, 'target', test_size=0.2, val_size=0.25)
    
    # Verificar se os resultados têm o formato esperado
    assert len(result) == 6  # Deve retornar 6 elementos
    
    X_train, X_val, X_test, y_train, y_val, y_test = result
    
    # Verificar se os objetos retornados são do tipo esperado
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, (pd.Series, np.ndarray))
    assert isinstance(y_val, (pd.Series, np.ndarray))
    assert isinstance(y_test, (pd.Series, np.ndarray))
    
    # Verificar se as dimensões fazem sentido (aproximadamente)
    n_samples = len(df)
    assert len(X_test) > 0 and len(X_test) < n_samples
    assert len(X_val) > 0 and len(X_val) < n_samples
    assert len(X_train) > 0 and len(X_train) < n_samples
    assert len(X_train) + len(X_val) + len(X_test) == n_samples
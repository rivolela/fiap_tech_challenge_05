"""
cross_validation.py - Implementação de técnicas de validação cruzada e prevenção de data leakage

Este módulo fornece funções para:
1. Realizar validação cruzada adequada (com k-fold)
2. Aplicar oversampling apenas nos dados de treinamento
3. Detectar e prevenir data leakage
4. Implementar feature selection para reduzir overfitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_selection import SelectFromModel, RFE, RFECV, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


def detect_leakage_candidates(df, target_col, threshold=0.9):
    """
    Detecta features que podem estar causando data leakage baseado em correlação com o target
    ou outros critérios suspeitos.
    
    Args:
        df: DataFrame com os dados
        target_col: Nome da coluna target
        threshold: Limite de correlação para considerar uma feature suspeita
    
    Returns:
        Lista de colunas potencialmente causando data leakage
    """
    leakage_candidates = []
    
    # 1. Verificar correlação com o target
    for col in df.columns:
        if col == target_col:
            continue
            
        # Para features categóricas, usamos chi2 ou correlação de Cramér
        if df[col].dtype == 'object' or df[col].dtype == 'category' or df[col].nunique() < 10:
            # Calcular correlação baseada em contagem
            contingency = pd.crosstab(df[col], df[target_col])
            
            # Verificar se há categorias com 100% de associação
            row_totals = contingency.sum(axis=1)
            for i, category in enumerate(contingency.index):
                for target_val in contingency.columns:
                    if contingency.iloc[i][target_val] == row_totals.iloc[i] and row_totals.iloc[i] > 10:
                        leakage_candidates.append((col, 
                                                 f"Categoria '{category}' tem 100% de associação com target={target_val}"))
        
        # Para features numéricas, usamos correlação padrão
        elif pd.api.types.is_numeric_dtype(df[col]):
            corr = df[[col, target_col]].corr().iloc[0, 1]
            if abs(corr) > threshold:
                leakage_candidates.append((col, f"Alta correlação com target: {corr:.3f}"))
    
    # 2. Verificar nomes de colunas suspeitas (específicos do problema)
    suspicious_terms = [
        'situacao', 'status', 'resultado', 'contratado', 'aprovado', 'sucesso', 
        'target', 'output', 'outcome', 'decision', 'final'
    ]
    
    for term in suspicious_terms:
        for col in df.columns:
            if term in col.lower() and col != target_col and col not in [c[0] for c in leakage_candidates]:
                leakage_candidates.append((col, f"Nome suspeito contendo '{term}'"))
    
    # 3. Verificar colunas com data futura ao processo de decisão
    date_cols = [col for col in df.columns if 'data' in col.lower()]
    for col in date_cols:
        if ('final' in col.lower() or 
            'contratacao' in col.lower() or 
            'resultado' in col.lower()):
            leakage_candidates.append((col, "Data possivelmente futura à decisão"))
    
    print(f"⚠️ Detectadas {len(leakage_candidates)} possíveis fontes de data leakage:")
    for col, reason in leakage_candidates:
        print(f"  - {col}: {reason}")
        
    return [col for col, _ in leakage_candidates]


def select_features(X, y, method='rfe', n_features=20):
    """
    Seleciona features mais importantes usando diferentes métodos
    
    Args:
        X: Features
        y: Target
        method: Método de seleção ('rfe', 'rfecv', 'selectkbest', 'model')
        n_features: Número de features a selecionar
        
    Returns:
        Lista de nomes das features selecionadas
    """
    feature_names = X.columns.tolist()
    
    if method == 'rfe':
        # Recursive Feature Elimination
        selector = RFE(
            estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            n_features_to_select=n_features
        )
        
    elif method == 'rfecv':
        # RFE with Cross Validation
        selector = RFECV(
            estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            min_features_to_select=n_features,
            cv=5
        )
        
    elif method == 'selectkbest':
        # Select K Best
        selector = SelectKBest(f_classif, k=n_features)
        
    elif method == 'model':
        # Feature importance based selection
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            max_features=n_features
        )
    
    else:
        raise ValueError(f"Método desconhecido: {method}")
    
    selector.fit(X, y)
    
    if method == 'selectkbest':
        selected_mask = selector.get_support()
    elif method == 'model':
        selected_mask = selector.get_support()
    else:
        selected_mask = selector.support_
        
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    
    print(f"✅ {len(selected_features)} features selecionadas usando método '{method}'")
    
    return selected_features


def perform_stratified_kfold_cv(X, y, model, n_splits=5):
    """
    Executa validação cruzada estratificada com k-fold, aplicando
    balanceamento separadamente em cada fold de treinamento
    
    Args:
        X: Features
        y: Target
        model: Modelo a ser avaliado
        n_splits: Número de folds
        
    Returns:
        Média e desvio padrão das métricas por fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Métricas por fold
    metrics = {
        'accuracy': [],
        'auc': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    print(f"🔄 Executando validação cruzada com {n_splits} folds...")
    
    from collections import Counter
    print(f"  - Distribuição original de classes: {dict(Counter(y))}")
    
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Balanceamento APENAS nos dados de treino deste fold
        X_resampled, y_resampled = balance_training_data(X_fold_train, y_fold_train)
        
        # Treinamento com dados balanceados
        model.fit(X_resampled, y_resampled)
        
        # Avaliação com dados não balanceados
        y_pred = model.predict(X_fold_val)
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
        
        # Coleta de métricas
        metrics['accuracy'].append(accuracy_score(y_fold_val, y_pred))
        metrics['precision'].append(precision_score(y_fold_val, y_pred))
        metrics['recall'].append(recall_score(y_fold_val, y_pred))
        metrics['f1'].append(f1_score(y_fold_val, y_pred))
        metrics['auc'].append(roc_auc_score(y_fold_val, y_pred_proba))
        
        print(f"  - Fold {i+1}: AUC={metrics['auc'][-1]:.4f}, F1={metrics['f1'][-1]:.4f}")
    
    # Calcular média e desvio padrão
    results = {}
    for metric, values in metrics.items():
        results[metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    # Exibir resultados
    print("\n📊 Resultados da validação cruzada:")
    for metric, stats in results.items():
        print(f"  - {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return results


def balance_training_data(X_train, y_train):
    """
    Balanceia os dados de treino usando oversampling simples da classe minoritária
    
    Args:
        X_train: Features de treinamento
        y_train: Target de treinamento
        
    Returns:
        X_resampled, y_resampled: Dados balanceados
    """
    # Identificar classes e suas contagens
    class_counts = pd.Series(y_train).value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    
    # Separar exemplos por classe
    minority_indices = np.where(y_train == minority_class)[0]
    majority_indices = np.where(y_train == majority_class)[0]
    
    # Oversample a classe minoritária
    minority_resampled = np.random.choice(
        minority_indices, 
        size=len(majority_indices), 
        replace=True
    )
    
    # Combinar exemplos
    combined_indices = np.concatenate([majority_indices, minority_resampled])
    
    # Recriar conjuntos de treino balanceados
    X_train_balanced = X_train.iloc[combined_indices].reset_index(drop=True)
    y_train_balanced = pd.Series(y_train).iloc[combined_indices].reset_index(drop=True)
    
    return X_train_balanced, y_train_balanced


def visualize_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Visualiza e salva as features mais importantes do modelo
    
    Args:
        model: Modelo treinado com feature_importances_
        feature_names: Lista com nomes das features
        top_n: Número de features a mostrar
        save_path: Caminho para salvar a visualização
    """
    # Verificar se o modelo tem feature_importances_
    if not hasattr(model, 'feature_importances_'):
        if hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier', None), 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
        else:
            print("⚠️ Modelo não possui feature_importances_")
            return
    else:
        importances = model.feature_importances_
    
    # Verificar se temos nomes de features suficientes
    if len(feature_names) < len(importances):
        print(f"⚠️ Número de nomes de features ({len(feature_names)}) menor que o número de importâncias ({len(importances)})")
        print("⚠️ Usando índices numéricos para features sem nome")
        # Estender a lista de nomes com índices para features sem nome
        feature_names = list(feature_names) + [f"Feature_{i}" for i in range(len(feature_names), len(importances))]
    
    # Verificar se temos nomes de features demais (para segurança)
    if len(feature_names) > len(importances):
        print(f"⚠️ Número de nomes de features ({len(feature_names)}) maior que o número de importâncias ({len(importances)})")
        print("⚠️ Usando apenas os primeiros nomes que correspondem às importâncias")
        feature_names = feature_names[:len(importances)]
    
    # Organizar as features por importância
    indices = np.argsort(importances)[::-1]
    
    # Garantir que top_n não seja maior que o número de features
    top_n = min(top_n, len(indices))
    
    # Selecionar as top_n features
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    
    # Criar visualização
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_importances)), top_importances, align='center')
    plt.yticks(range(len(top_importances)), top_features)
    plt.xlabel('Importância Relativa')
    plt.ylabel('Features')
    plt.title(f'Top {top_n} Features Mais Importantes')
    plt.tight_layout()
    
    # Salvar ou mostrar
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ Gráfico de importância de features salvo em: {save_path}")
    else:
        plt.show()
        
    # Retornar as features importantes para análise
    importance_df = pd.DataFrame({
        'feature': top_features,
        'importance': top_importances
    }).sort_values('importance', ascending=False)
    
    return importance_df


def safe_split_and_balance(df, target_col, test_size=0.2, val_size=0.25, random_state=42):
    """
    Realiza a divisão dos dados e balanceamento de forma segura,
    aplicando o balanceamento APENAS após a separação de train/val/test
    
    Args:
        df: DataFrame com os dados
        target_col: Nome da coluna target
        test_size: Proporção dos dados para teste
        val_size: Proporção dos dados restantes para validação
        random_state: Semente aleatória
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Preparar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Primeira divisão: separa dados de teste
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Segunda divisão: separa dados de treino e validação
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    
    print(f"\n🔄 Divisão segura de dados:")
    print(f"  - Conjunto de treino: {X_train.shape[0]} amostras")
    print(f"  - Conjunto de validação: {X_val.shape[0]} amostras")
    print(f"  - Conjunto de teste: {X_test.shape[0]} amostras")
    
    # Verificar distribuição das classes em cada conjunto
    for name, y_set in [('treino', y_train), ('validação', y_val), ('teste', y_test)]:
        class_dist = pd.Series(y_set).value_counts(normalize=True) * 100
        print(f"\n📊 Distribuição de classes ({name}):")
        for cls, pct in class_dist.items():
            print(f"  - Classe {cls}: {pct:.1f}%")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Exemplo de uso
    print("Módulo de validação cruzada e prevenção de data leakage")
    print("Use 'import cross_validation' para utilizar estas funções em outros módulos")

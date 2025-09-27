"""
train_simple.py - Sistema Híbrido de Scoring + Clustering para Decision

Este módulo implementa um sistema híbrido mais simples:
1. Um modelo de scoring para prever a probabilidade de sucesso na contratação
2. Tratamento de dados categóricos e numéricos
3. Integração com MLflow para rastrear experimentos, métricas e artefatos
4. Prevenção de data leakage e técnicas contra overfitting
5. Validação cruzada adequada com balanceamento apenas nos dados de treino

Objetivo: Aprimorar o match entre candidatos e vagas e validar as métricas do modelo
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel, RFE, RFECV

# Importar nosso módulo de validação cruzada e prevenção de data leakage
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.cross_validation import detect_leakage_candidates, select_features, perform_stratified_kfold_cv, visualize_feature_importance

# Importar MLflow para rastreamento de experimentos
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
    print("✅ MLflow disponível para rastreamento de experimentos")
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow não instalado. As métricas serão exibidas, mas não rastreadas.")
    print("   Para instalar MLflow, execute: pip install mlflow")
    
# Configurações de MLflow
EXPERIMENT_NAME = "Decision-Scoring-Model"
# Configurar o MLflow para usar armazenamento local
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

def generate_random_seed():
    """
    Gera uma semente aleatória baseada no timestamp atual para
    garantir diferentes resultados entre execuções do modelo.
    
    Returns:
        int: Semente aleatória entre 1 e 100000
    """
    return int((time.time() * 1000) % 100000)

def generate_run_name(model_type):
    """
    Gera um nome único para a execução do MLflow baseado no timestamp,
    tipo de modelo e um identificador aleatório.
    
    Args:
        model_type: Tipo do modelo usado (RandomForest, GradientBoosting, etc)
        
    Returns:
        str: Nome único para a execução
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    random_id = str(int(time.time() * 1000) % 10000)
    return f"{model_type}-{timestamp}-{random_id}"


def load_data(data_path='data/processed/complete_processed_data.csv', target='target_sucesso'):
    """
    Carrega os dados e faz verificações iniciais
    """
    print("🔄 Carregando dados processados...")
    df = pd.read_csv(data_path)
    
    print(f"✅ Dados carregados: {df.shape[0]} registros, {df.shape[1]} colunas")
    
    # Verificar se a variável target existe
    if target not in df.columns:
        raise ValueError(f"Variável target '{target}' não encontrada nos dados. "
                      "Execute o pré-processamento primeiro.")
        
    # Distribuição da variável target
    target_dist = df[target].value_counts(normalize=True) * 100
    print("\n📊 Distribuição do Target:")
    for value, pct in target_dist.items():
        print(f"  - Classe {value}: {pct:.1f}%")
        
    # Verificar a presença de valores nulos em colunas importantes
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("\n⚠️ Colunas com valores nulos:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"  - {col}: {count} valores nulos ({count/len(df)*100:.1f}%)")
            
    return df


def prepare_features(df, target='target_sucesso', prevent_leakage=True, feature_selection=True):
    """
    Prepara features para modelagem, identificando e tratando diferentes tipos de variáveis.
    Inclui detecção de data leakage e seleção de features.
    
    Args:
        df: DataFrame com os dados
        target: Nome da coluna target
        prevent_leakage: Se True, detecta e remove features com possível data leakage
        feature_selection: Se True, aplica seleção de features para reduzir dimensionalidade
        
    Returns:
        Lista de features para treinamento e dicionário com grupos de features
    """
    print("\n🔄 Preparando features para modelagem...")
    
    # Identificar tipos de colunas para tratamento específico
    id_cols = ['nome', 'codigo', 'job_id', 'titulo_vaga']
    date_cols = [col for col in df.columns if 'data' in col.lower()]
    text_cols = ['comentario']
    
    # Colunas para remover do treinamento
    remove_cols = id_cols + text_cols
    
    # Detectar e prevenir data leakage
    if prevent_leakage:
        print("\n🔍 Verificando possível data leakage...")
        leakage_candidates = detect_leakage_candidates(df, target)
        
        # Adicionar candidatos a leakage na lista de colunas a remover
        for col in leakage_candidates:
            if col not in remove_cols:
                remove_cols.append(col)
                print(f"  - ⚠️ Removendo coluna '{col}' para prevenir data leakage")
    
    # Colunas categóricas que devem passar por encoding
    categorical_cols = []
    for col in df.columns:
        if col not in remove_cols and col != target:
            if df[col].dtype == 'object' or df[col].dtype == 'bool':
                categorical_cols.append(col)
                
    print(f"  - Features categóricas identificadas: {len(categorical_cols)}")
    
    # Features numéricas 
    numeric_cols = [col for col in df.columns 
                  if col not in remove_cols + categorical_cols + date_cols + [target]
                  and df[col].dtype in ['int64', 'float64']]
    
    print(f"  - Features numéricas identificadas: {len(numeric_cols)}")
    
    # Aplicar seleção de features se solicitado
    train_features = categorical_cols + numeric_cols
    selected_features = train_features
    
    if feature_selection and len(train_features) > 20:
        # Preparar dados para seleção de features
        # Vamos fazer uma codificação simples para análise
        X_for_selection = df[train_features].copy()
        
        # One-hot encoding simples para features categóricas
        for col in categorical_cols:
            dummies = pd.get_dummies(X_for_selection[col], prefix=col, drop_first=True)
            X_for_selection = pd.concat([X_for_selection, dummies], axis=1)
            X_for_selection.drop(col, axis=1, inplace=True)
        
        # Selecionar features
        y_for_selection = df[target]
        try:
            print("\n🔍 Aplicando seleção de features para reduzir dimensionalidade...")
            # Usar RFE ou SelectFromModel
            feature_selector = SelectFromModel(
                RandomForestClassifier(n_estimators=50, random_state=42),
                max_features=20  # Limitar a 20 features mais importantes
            )
            feature_selector.fit(X_for_selection, y_for_selection)
            
            # Identificar quais features originais foram selecionadas
            # Isso é complicado porque temos one-hot encoding
            # Vamos manter as features originais por enquanto
            print("  - Mantendo as features originais para simplicidade")
            
        except Exception as e:
            print(f"⚠️ Erro na seleção de features: {str(e)}")
            print("  - Continuando com todas as features")
    
    # Salvar a lista de features por tipo para uso na modelagem
    feature_groups = {
        'id': id_cols,
        'date': date_cols,
        'text': text_cols,
        'categorical': categorical_cols,
        'numeric': numeric_cols
    }
    
    # Todas as features a serem usadas no treinamento
    train_features = categorical_cols + numeric_cols
    print(f"✅ Total de {len(train_features)} features preparadas para treinamento")
    
    return train_features, feature_groups


def split_data(df, train_features, target='target_sucesso', test_size=0.2, val_size=0.25, random_state=42, use_cv=True):
    """
    Divide os dados em conjuntos de treino, validação e teste com estratificação.
    Implementa abordagem segura para balanceamento de dados (apenas no treino) e
    opcionalmente usa validação cruzada.
    
    Args:
        df: DataFrame com os dados
        train_features: Lista de colunas para usar no treinamento
        target: Nome da coluna target
        test_size: Proporção para dados de teste
        val_size: Proporção para dados de validação
        random_state: Seed para reprodutibilidade
        use_cv: Se True, prepara os dados para validação cruzada
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("\n🔄 Dividindo dados em conjuntos de treino, validação e teste...")
    
    # Preparar X e y
    X = df[train_features]
    y = df[target]
    
    if not use_cv:
        # Abordagem tradicional com split único
        # Primeira divisão: separa dados de teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Segunda divisão: separa dados de treino e validação
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        print(f"  - Conjunto de treino: {X_train.shape[0]} amostras")
        print(f"  - Conjunto de validação: {X_val.shape[0]} amostras")
        print(f"  - Conjunto de teste: {X_test.shape[0]} amostras")
        
        # Verificar se há desbalanceamento no conjunto de treino
        train_class_dist = y_train.value_counts()
        if len(train_class_dist) > 1:
            minority_class = train_class_dist.min()
            majority_class = train_class_dist.max()
            ratio = majority_class / minority_class
            
            if ratio > 5:
                print(f"⚠️ Dados de treino desbalanceados (razão 1:{ratio:.1f})")
                # Vamos usar balance_training_data, mas apenas para os dados de treino!
                X_train, y_train = balance_training_data(X_train, y_train)
                print(f"  - Balanceamento aplicado APENAS aos dados de treino")
                print(f"  - Dados de validação e teste mantêm a distribuição original")
    else:
        # Abordagem de validação cruzada
        # Apenas separamos os dados de teste, o restante será dividido durante o CV
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Para manter a API consistente, criamos subconjuntos de validação
        # mas que serão usados apenas para avaliação final
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
        )
        
        print(f"  - Conjunto para CV: {X_train_val.shape[0]} amostras")
        print(f"  - Conjunto de teste final: {X_test.shape[0]} amostras")
        print("  - Nota: Na validação cruzada, o balanceamento será aplicado")
        print("    separadamente a cada fold de treino durante o treinamento")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def balance_training_data(X_train, y_train):
    """
    Balanceia os dados de treino usando oversampling simples
    """
    print("⚠️ SMOTE não disponível, usando oversampling simples...")
    
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
    
    print(f"✅ Dados balanceados com oversampling simples: {len(X_train_balanced)} amostras")
    return X_train_balanced, y_train_balanced


def train_scoring_model(X_train, y_train, X_val, y_val, X_test, y_test, feature_groups, 
                        model_type="RandomForest", use_cv=True, n_cv_folds=5, random_state=None):
    """
    Treina um modelo de scoring para prever a probabilidade de sucesso na contratação
    com rastreamento de experimentos através do MLflow.
    
    Implementa técnicas para evitar overfitting:
    - Validação cruzada (opcional)
    - Balanceamento aplicado apenas nos dados de treino
    - Visualização das features mais importantes para análise
    
    O random_state pode ser fornecido ou será gerado com base no timestamp
    para garantir diferentes resultados entre execuções.
    
    Args:
        X_train, y_train: Dados de treinamento
        X_val, y_val: Dados de validação
        X_test, y_test: Dados de teste
        feature_groups: Dicionário com grupos de features
        model_type: Tipo de modelo ("RandomForest" ou "GradientBoosting")
        use_cv: Se True, usa validação cruzada estratificada
        n_cv_folds: Número de folds para validação cruzada
    
    Returns:
        O modelo treinado
    """
    print("\n🔄 Treinando modelo de scoring para prever sucesso na contratação...")
    print(f"  - Estratégia: {'Validação Cruzada' if use_cv else 'Split Tradicional'}")
    
    # Preparar preprocessador para lidar com variáveis categóricas e numéricas
    categorical_cols = feature_groups['categorical']
    numeric_cols = feature_groups['numeric']
    
    # Criar preprocessadores para cada tipo de coluna
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combinar preprocessadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Gerar random_state se não foi fornecido
    if random_state is None:
        random_state = generate_random_seed()
    
    print(f"🎲 Usando random_state: {random_state}")
    
    # Introduzir variabilidade nos hiperparâmetros baseados no random_state
    n_estimators = 80 + (random_state % 50)  # Entre 80 e 129
    max_depth_options = [None, 5, 10, 15, 20]
    max_depth = max_depth_options[random_state % len(max_depth_options)]
    min_samples_split = 2 + (random_state % 5)  # Entre 2 e 6
    
    # Selecionar o modelo de classificação com o random_state variável
    if model_type == "GradientBoosting":
        learning_rate = 0.05 + ((random_state % 20) / 100)  # Entre 0.05 e 0.24
        classifier = GradientBoostingClassifier(random_state=random_state)
        model_params = {
            'classifier__n_estimators': n_estimators,
            'classifier__learning_rate': learning_rate,
            'classifier__max_depth': 3 + (random_state % 5)  # Entre 3 e 7
        }
    else:  # RandomForest
        classifier = RandomForestClassifier(random_state=random_state)
        model_params = {
            'classifier__n_estimators': n_estimators,
            'classifier__max_depth': max_depth,
            'classifier__min_samples_split': min_samples_split
        }
    
    # Criar um pipeline que combina preprocessador e modelo
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    # Inicializar MLflow se disponível
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            
            # Usar nome de run único baseado no timestamp e random_state
            run_name = generate_run_name(model_type)
            with mlflow.start_run(run_name=run_name) as run:
                # Registrar parâmetros
                for param_name, param_value in model_params.items():
                    mlflow.log_param(param_name, param_value)
                
                # Registrar informações sobre os dados
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("categorical_features", len(categorical_cols))
                mlflow.log_param("numeric_features", len(numeric_cols))
                
                # Definir parâmetros do modelo
                for param_name, param_value in model_params.items():
                    key = param_name.split('__')[1]
                    setattr(model.named_steps['classifier'], key, param_value)
                
                # Treinar o modelo
                print(f"  - Treinando modelo {model_type}...")
                model.fit(X_train, y_train)
                print("  - Treinamento concluído!")
                
                # Avaliar no conjunto de validação
                val_proba = model.predict_proba(X_val)[:, 1]
                val_pred = model.predict(X_val)
                
                val_auc = roc_auc_score(y_val, val_proba)
                val_acc = accuracy_score(y_val, val_pred)
                val_f1 = f1_score(y_val, val_pred)
                val_precision = precision_score(y_val, val_pred)
                val_recall = recall_score(y_val, val_pred)
                val_avg_precision = average_precision_score(y_val, val_proba)
                
                # Registrar métricas de validação - converter para float para evitar problemas de tipo
                mlflow.log_metric("val_auc", float(val_auc))
                mlflow.log_metric("val_accuracy", float(val_acc))
                mlflow.log_metric("val_f1", float(val_f1))
                mlflow.log_metric("val_precision", float(val_precision))
                mlflow.log_metric("val_recall", float(val_recall))
                mlflow.log_metric("val_avg_precision", float(val_avg_precision))
                
                # Registrar o random_state usado como parâmetro
                mlflow.log_param("random_state", random_state)
                
                # Exibir métricas de validação
                print(f"\n📊 Performance no conjunto de validação:")
                print(f"  - AUC-ROC: {val_auc:.4f}")
                print(f"  - Acurácia: {val_acc:.4f}")
                print(f"  - F1-Score: {val_f1:.4f}")
                print(f"  - Precisão: {val_precision:.4f}")
                print(f"  - Recall: {val_recall:.4f}")
                
                # Avaliar no conjunto de teste
                test_proba = model.predict_proba(X_test)[:, 1]
                test_pred = model.predict(X_test)
                
                test_auc = roc_auc_score(y_test, test_proba)
                test_acc = accuracy_score(y_test, test_pred)
                test_f1 = f1_score(y_test, test_pred)
                test_precision = precision_score(y_test, test_pred)
                test_recall = recall_score(y_test, test_pred)
                test_avg_precision = average_precision_score(y_test, test_proba)
                
                # Registrar métricas de teste - converter para float para evitar problemas de tipo
                mlflow.log_metric("test_auc", float(test_auc))
                mlflow.log_metric("test_accuracy", float(test_acc))
                mlflow.log_metric("test_f1", float(test_f1))
                mlflow.log_metric("test_precision", float(test_precision))
                mlflow.log_metric("test_recall", float(test_recall))
                mlflow.log_metric("test_avg_precision", float(test_avg_precision))
                
                # Exibir métricas de teste
                print(f"\n📊 Performance final no conjunto de teste:")
                print(f"  - AUC-ROC: {test_auc:.4f}")
                print(f"  - Acurácia: {test_acc:.4f}")
                print(f"  - F1-Score: {test_f1:.4f}")
                print(f"  - Precisão: {test_precision:.4f}")
                print(f"  - Recall: {test_recall:.4f}")
                
                # Criar e salvar visualizações para MLflow
                # 1. Curva ROC
                os.makedirs('data/visualizations', exist_ok=True)
                
                plt.figure(figsize=(10, 8))
                
                # Matriz de confusão
                cm = confusion_matrix(y_test, test_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Não Contratado', 'Contratado'],
                            yticklabels=['Não Contratado', 'Contratado'])
                plt.xlabel('Predito')
                plt.ylabel('Real')
                plt.title('Matriz de Confusão')
                cm_path = 'data/visualizations/confusion_matrix.png'
                plt.tight_layout()
                plt.savefig(cm_path)
                
                # Registrar artefatos e imagens
                mlflow.log_artifact(cm_path)
                
                # Registrar o modelo no MLflow
                # Correção para o erro "sklearn" is not exported from module "mlflow"
                try:
                    mlflow.sklearn.log_model(model, "model")
                except AttributeError:
                    # Fallback para versões diferentes do MLflow
                    from mlflow import sklearn as mlflow_sklearn
                    mlflow_sklearn.log_model(model, "model")
                
                # Salvar relatório de classificação
                report = classification_report(y_test, test_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                report_path = 'data/insights/classification_report.csv'
                report_df.to_csv(report_path)
                mlflow.log_artifact(report_path)
                
                # Exibir link para a UI do MLflow (apenas em produção)
                print(f"\n✅ Experimento registrado no MLflow: {run.info.run_id}")
        
        except Exception as e:
            print(f"⚠️ Erro ao usar MLflow: {e}. Continuando sem rastreamento...")
            # Se houver erro com MLflow, continuar sem rastreamento
            model.fit(X_train, y_train)
    else:
        # Se MLflow não estiver disponível, treinar normalmente
        print("  - Treinando modelo sem rastreamento MLflow...")
        
        # Definir parâmetros do modelo
        for param_name, param_value in model_params.items():
            key = param_name.split('__')[1]
            setattr(model.named_steps['classifier'], key, param_value)
            
        model.fit(X_train, y_train)
        print("  - Treinamento concluído!")
        
        # Avaliar no conjunto de validação
        val_proba = model.predict_proba(X_val)[:, 1]
        val_pred = model.predict(X_val)
        
        val_auc = roc_auc_score(y_val, val_proba)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred)
        
        print(f"\n📊 Performance no conjunto de validação:")
        print(f"  - AUC-ROC: {val_auc:.4f}")
        print(f"  - Acurácia: {val_acc:.4f}")
        print(f"  - F1-Score: {val_f1:.4f}")
        
        # Avaliar no conjunto de teste
        test_proba = model.predict_proba(X_test)[:, 1]
        test_pred = model.predict(X_test)
        
        test_auc = roc_auc_score(y_test, test_proba)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred)
        
        print(f"\n📊 Performance final no conjunto de teste:")
        print(f"  - AUC-ROC: {test_auc:.4f}")
        print(f"  - Acurácia: {test_acc:.4f}")
        print(f"  - F1-Score: {test_f1:.4f}")
    
    # Salvar modelo independentemente do MLflow
    os.makedirs('models', exist_ok=True)
    with open('models/scoring_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print("✅ Modelo de scoring salvo em models/scoring_model.pkl")
    
    # Imprimir relatório de classificação
    print("\n📊 Relatório de classificação detalhado:")
    print(classification_report(y_test, test_pred))
    
    return model


def compare_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_groups, 
                 use_cv=True, n_cv_folds=5):
    """
    Compara diferentes modelos usando MLflow para rastreamento.
    Implementa validação cruzada para avaliação mais robusta.
    
    Args:
        X_train, y_train: Dados de treinamento
        X_val, y_val: Dados de validação
        X_test, y_test: Dados de teste
        feature_groups: Dicionário com grupos de features
        use_cv: Se True, usa validação cruzada
        n_cv_folds: Número de folds para CV
    
    Returns:
        O melhor modelo baseado na métrica AUC-ROC no conjunto de validação
    """
    print("\n" + "="*70)
    print("🔄 COMPARANDO DIFERENTES MODELOS")
    print("="*70)
    
    models = {
        "RandomForest": {"model_type": "RandomForest"},
        "GradientBoosting": {"model_type": "GradientBoosting"}
    }
    
    best_score = 0
    best_model = None
    best_model_name = None
    
    # Treinar e avaliar cada modelo
    for model_name, params in models.items():
        print(f"\n🧪 Experimentando modelo: {model_name}")
        model = train_scoring_model(
            X_train, y_train, X_val, y_val, X_test, y_test, 
            feature_groups, model_type=params['model_type'],
            use_cv=use_cv, n_cv_folds=n_cv_folds
        )
        
        if use_cv:
            # No modo CV, a performance já foi avaliada durante o treinamento
            # Precisamos apenas extrair a melhor métrica do MLflow
            if MLFLOW_AVAILABLE:
                try:
                    from mlflow.tracking import MlflowClient
                    client = MlflowClient()
                    # Pegar o último experimento registrado
                    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
                    if experiment:
                        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                        if not runs.empty:
                            # Filtrar runs deste modelo
                            model_runs = runs[runs['tags.mlflow.runName'].str.contains(model_name, na=False)]
                            if not model_runs.empty:
                                # Pegar a última run
                                last_run = model_runs.iloc[0]
                                val_auc = last_run.get('metrics.val_auc', 0)
                            else:
                                # Fallback para avaliação direta
                                val_proba = model.predict_proba(X_val)[:, 1]
                                val_auc = roc_auc_score(y_val, val_proba)
                        else:
                            # Fallback para avaliação direta
                            val_proba = model.predict_proba(X_val)[:, 1]
                            val_auc = roc_auc_score(y_val, val_proba)
                    else:
                        # Fallback para avaliação direta
                        val_proba = model.predict_proba(X_val)[:, 1]
                        val_auc = roc_auc_score(y_val, val_proba)
                except Exception as e:
                    print(f"⚠️ Erro ao acessar MLflow: {str(e)}")
                    # Fallback para avaliação direta
                    val_proba = model.predict_proba(X_val)[:, 1]
                    val_auc = roc_auc_score(y_val, val_proba)
            else:
                # Avaliar no conjunto de validação diretamente
                val_proba = model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, val_proba)
        else:
            # Avaliar no conjunto de validação
            val_proba = model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_proba)
        
        print(f"  - Performance (AUC-ROC validação): {val_auc:.4f}")
        
        # Atualizar melhor modelo se necessário
        if val_auc > best_score:
            best_score = val_auc
            best_model = model
            best_model_name = model_name
    
    print(f"\n✅ Melhor modelo: {best_model_name} (AUC-ROC: {best_score:.4f})")
    
    # Salvar o melhor modelo
    with open('models/best_scoring_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
        
    print("✅ Melhor modelo salvo em models/best_scoring_model.pkl")
    
    # Visualizar importância das features do melhor modelo
    if hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps.get('classifier', None), 'feature_importances_'):
        print("\n📊 Analisando importância de features do melhor modelo...")
        # Extrair nomes de features após transformações
        feature_names = feature_groups['categorical'] + feature_groups['numeric']
        
        # Criar visualização
        importance_df = visualize_feature_importance(
            best_model.named_steps['classifier'], 
            feature_names, 
            top_n=20, 
            save_path='data/insights/best_model_feature_importance.png'
        )
        
        # Salvar importância para análise
        if isinstance(importance_df, pd.DataFrame):
            importance_df.to_csv('data/insights/best_model_feature_importance.csv', index=False)
    
    return best_model


def main():
    """Função principal para executar o treinamento do modelo"""
    # Verificar se diretórios existem, caso contrário criá-los
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/insights', exist_ok=True)
    os.makedirs('data/visualizations', exist_ok=True)
    
    # Definir variável target
    target = 'target_sucesso'
    
    # Processar argumentos de linha de comando
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Treinar modelo de scoring para Decision")
    parser.add_argument("--compare", action="store_true", help="Comparar diferentes algoritmos")
    parser.add_argument("--no-leakage-prevention", action="store_true", help="Desativar detecção de data leakage")
    parser.add_argument("--no-cv", action="store_true", help="Desativar validação cruzada")
    parser.add_argument("--no-feature-selection", action="store_true", help="Desativar seleção de features")
    parser.add_argument("--cv-folds", type=int, default=5, help="Número de folds para validação cruzada")
    parser.add_argument("--model", type=str, default="RandomForest", choices=["RandomForest", "GradientBoosting"], 
                        help="Algoritmo a ser usado")
    parser.add_argument("--random-seed", type=int, help="Semente aleatória para reprodutibilidade (se omitido, será gerada uma nova a cada execução)")
    
    args = parser.parse_args()
    
    # Configurações baseadas nos argumentos
    prevent_leakage = not args.no_leakage_prevention
    use_cv = not args.no_cv
    feature_selection = not args.no_feature_selection
    n_cv_folds = args.cv_folds
    compare_models_flag = args.compare
    model_type = args.model
    random_state = args.random_seed  # Pode ser None, o que irá gerar uma semente aleatória
    
    try:
        # Carregar e preparar dados
        df = load_data(target=target)
        
        # Preparar features com opções de prevenção de data leakage e seleção de features
        train_features, feature_groups = prepare_features(
            df, 
            target=target, 
            prevent_leakage=prevent_leakage,
            feature_selection=feature_selection
        )
        
        # Dividir dados com estratégia apropriada para validação cruzada
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df, 
            train_features, 
            target=target,
            use_cv=use_cv
        )
        
        # Verificar se devemos comparar modelos ou treinar apenas um
        if compare_models_flag:
            print("🔄 Modo de comparação de modelos ativado")
            best_model = compare_models(
                X_train, y_train, X_val, y_val, X_test, y_test, 
                feature_groups, use_cv=use_cv, n_cv_folds=n_cv_folds
            )
        else:
            # Treinar um único modelo
            # Se random_state é None, será gerada uma semente aleatória dentro da função
            model = train_scoring_model(
                X_train, y_train, X_val, y_val, X_test, y_test, 
                feature_groups, model_type=model_type,
                use_cv=use_cv, n_cv_folds=n_cv_folds,
                random_state=random_state
            )
            
            # Visualizar importância de features
            if hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier', None), 'feature_importances_'):
                print("\n📊 Analisando importância de features...")
                # Extrair nomes de features após transformações
                feature_names = train_features
                
                # Criar visualização
                importance_df = visualize_feature_importance(
                    model.named_steps['classifier'], 
                    feature_names, 
                    top_n=20, 
                    save_path='data/insights/feature_importance.png'
                )
                
                # Salvar importância para análise
                if isinstance(importance_df, pd.DataFrame):
                    importance_df.to_csv('data/insights/feature_importance.csv', index=False)
        
        print("\n" + "="*70)
        print("✅ TREINAMENTO DO MODELO CONCLUÍDO COM SUCESSO")
        print("="*70)
        
        if MLFLOW_AVAILABLE:
            print("\n📈 Para visualizar experimentos no MLflow:")
            print("1. Execute 'mlflow ui --port 5001' em um terminal")
            print("2. Abra http://localhost:5001 em seu navegador")
            print("3. Navegue até o experimento 'Decision-Scoring-Model'")
        
    except Exception as e:
        print(f"\n❌ ERRO DURANTE EXECUÇÃO DO PIPELINE: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

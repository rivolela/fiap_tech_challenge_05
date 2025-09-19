"""
train_model.py - Sistema Híbrido de Scoring + Clustering para Decision

Este módulo implementa um sistema híbrido que combina:
1. Um modelo de scoring para prever a probabilidade de sucesso na contratação
2. Um modelo de clustering para agrupar candidatos e vagas com características similares
3. Um sistema de recomendação para melhorar o match entre candidatos e vagas

Objetivo: Aprimorar o match entre candidatos e vagas, resolvendo desafios como:
- Falta de padronização em entrevistas
- Dificuldade em identificar engajamento real de candidatos
- Etapas de entrevista puladas para agilizar o processo
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import zscore

# Tentar importar SMOTE para balanceamento, mas não falhar se não estiver disponível
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


class ModelTrainer:
    def __init__(self, data_path='data/processed/complete_processed_data.csv'):
        """
        Inicializa o treinador de modelo com o caminho para o arquivo de dados processados
        
        Args:
            data_path: Caminho para o arquivo de dados processados completo
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scoring_model = None
        self.clustering_model = None
        self.feature_importances = None
        self.candidate_clusters = None
        self.job_clusters = None
        
        # Configurações
        self.target = 'target_sucesso'  # Usamos a variável já criada no pre-processamento
        self.n_clusters = 5  # Número de clusters para agrupar candidatos/vagas
        
        # Criar diretórios necessários
        os.makedirs('models', exist_ok=True)
        os.makedirs('data/insights', exist_ok=True)
        os.makedirs('data/visualizations', exist_ok=True)
        
    def load_data(self):
        """Carrega e realiza verificações iniciais nos dados"""
        print("🔄 Carregando dados processados...")
        self.df = pd.read_csv(self.data_path)
        
        print(f"✅ Dados carregados: {self.df.shape[0]} registros, {self.df.shape[1]} colunas")
        
        # Verificar se a variável target existe
        if self.target not in self.df.columns:
            raise ValueError(f"Variável target '{self.target}' não encontrada nos dados. "
                           "Execute o pré-processamento primeiro.")
            
        # Distribuição da variável target
        target_dist = self.df[self.target].value_counts(normalize=True) * 100
        print("\n📊 Distribuição do Target:")
        for value, pct in target_dist.items():
            print(f"  - Classe {value}: {pct:.1f}%")
            
        # Verificar a presença de valores nulos em colunas importantes
        null_counts = self.df.isnull().sum()
        if null_counts.sum() > 0:
            print("\n⚠️ Colunas com valores nulos:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"  - {col}: {count} valores nulos ({count/len(self.df)*100:.1f}%)")
                
        return self.df
        
    def prepare_features(self):
        """Prepara features para modelagem, identificando e tratando diferentes tipos de variáveis"""
        print("\n🔄 Preparando features para modelagem...")
        
        # Identificar tipos de colunas para tratamento específico
        id_cols = ['nome', 'codigo', 'job_id', 'titulo_vaga']
        date_cols = [col for col in self.df.columns if 'data' in col.lower()]
        text_cols = ['comentario']
        
        # Colunas para remover do treinamento
        remove_cols = id_cols + text_cols
        
        # Colunas categóricas que devem passar por encoding (se ainda não tiverem sido)
        categorical_cols = []
        for col in self.df.columns:
            if col not in remove_cols and col != self.target:
                if self.df[col].dtype == 'object' or self.df[col].dtype == 'bool':
                    categorical_cols.append(col)
                
        print(f"  - Features categóricas identificadas: {len(categorical_cols)}")
        print("    Exemplos de valores categóricos:")
        for col in categorical_cols[:5]:  # Mostrar apenas os primeiros 5 para não poluir o output
            print(f"    - {col}: {self.df[col].value_counts().head(3).to_dict()}")
        
        # Features numéricas 
        numeric_cols = [col for col in self.df.columns 
                      if col not in remove_cols + categorical_cols + date_cols + [self.target]
                      and self.df[col].dtype in ['int64', 'float64']]
        
        print(f"  - Features numéricas identificadas: {len(numeric_cols)}")
        print("    Exemplos de features numéricas:")
        for col in numeric_cols[:5]:  # Mostrar apenas os primeiros 5
            print(f"    - {col}: min={self.df[col].min()}, max={self.df[col].max()}, mean={self.df[col].mean():.2f}")
        
        # Salvar a lista de features por tipo para uso na modelagem
        self.feature_groups = {
            'id': id_cols,
            'date': date_cols,
            'text': text_cols,
            'categorical': categorical_cols,
            'numeric': numeric_cols
        }
        
        # Todas as features a serem usadas no treinamento
        self.train_features = categorical_cols + numeric_cols
        print(f"✅ Total de {len(self.train_features)} features preparadas para treinamento")
        
        return self.train_features
        
    def split_data(self, test_size=0.2, val_size=0.25, random_state=42):
        """
        Divide os dados em conjuntos de treino, validação e teste com estratificação
        
        Args:
            test_size: Proporção dos dados para teste
            val_size: Proporção dos dados restantes para validação
            random_state: Semente para reprodutibilidade
        """
        print("\n🔄 Dividindo dados em conjuntos de treino, validação e teste...")
        
        # Preparar X e y
        X = self.df[self.train_features]
        y = self.df[self.target]
        
        # Primeira divisão: separa dados de teste
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Segunda divisão: separa dados de treino e validação
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        print(f"  - Conjunto de treino: {self.X_train.shape[0]} amostras")
        print(f"  - Conjunto de validação: {self.X_val.shape[0]} amostras")
        print(f"  - Conjunto de teste: {self.X_test.shape[0]} amostras")
        
        # Verificar se há desbalanceamento no conjunto de treino
        train_class_dist = pd.Series(self.y_train).value_counts()
        if len(train_class_dist) > 1:
            minority_class = train_class_dist.min()
            majority_class = train_class_dist.max()
            ratio = majority_class / minority_class
            
            if ratio > 5:
                print(f"⚠️ Dados de treino desbalanceados (razão 1:{ratio:.1f})")
                self._balance_training_data()
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def _balance_training_data(self):
        """Balanceia os dados de treino usando SMOTE ou oversampling simples"""
        if SMOTE_AVAILABLE:
            try:
                print("🔄 Aplicando SMOTE para balancear os dados de treino...")
                smote = SMOTE(random_state=42)
                self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
                print(f"✅ Dados balanceados com SMOTE: {self.X_train.shape[0]} amostras")
            except Exception as e:
                print(f"⚠️ Erro ao aplicar SMOTE: {e}")
                self._apply_simple_oversampling()
        else:
            print("⚠️ SMOTE não disponível, usando oversampling simples...")
            self._apply_simple_oversampling()
    
    def _apply_simple_oversampling(self):
        """Aplica oversampling simples para balancear os dados"""
        # Identificar classes e suas contagens
        class_counts = pd.Series(self.y_train).value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        
        # Separar exemplos por classe
        minority_indices = np.where(self.y_train == minority_class)[0]
        majority_indices = np.where(self.y_train == majority_class)[0]
        
        # Oversample a classe minoritária
        minority_resampled = np.random.choice(
            minority_indices, 
            size=len(majority_indices), 
            replace=True
        )
        
        # Combinar exemplos
        combined_indices = np.concatenate([majority_indices, minority_resampled])
        
        # Recriar conjuntos de treino balanceados
        self.X_train = self.X_train.iloc[combined_indices].reset_index(drop=True)
        self.y_train = pd.Series(self.y_train).iloc[combined_indices].reset_index(drop=True)
        
        print(f"✅ Dados balanceados com oversampling simples: {len(self.X_train)} amostras")
    
    def train_scoring_model(self):
        """Treina o modelo de scoring para prever probabilidade de sucesso na contratação"""
        print("\n🔄 Treinando modelo de scoring para prever sucesso na contratação...")
        
        # Preparar preprocessador para lidar com variáveis categóricas e numéricas
        categorical_cols = self.feature_groups['categorical']
        numeric_cols = self.feature_groups['numeric']
        
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
        
        # Definir modelos candidatos para o Grid Search com pipeline de preprocessamento
        models = {
            'RandomForest': Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=42))
            ]),
            'GradientBoosting': Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(random_state=42))
            ])
        }
        
        # Parâmetros para Grid Search
        param_grids = {
            'RandomForest': {
                'classifier__n_estimators': [100],
                'classifier__max_depth': [None, 10],
                'classifier__min_samples_split': [2, 5]
            },
            'GradientBoosting': {
                'classifier__n_estimators': [100],
                'classifier__max_depth': [3, 5],
                'classifier__learning_rate': [0.1]
            }
        }
        
        best_score = 0
        best_model = None
        best_model_name = None
        
        # Avaliar cada modelo com validação cruzada
        for model_name, model in models.items():
            print(f"\n📊 Avaliando modelo: {model_name}")
            
            # Configurar Grid Search com validação cruzada
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[model_name],
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1
            )
            
            # Treinar modelo
            try:
                print(f"  - Iniciando treinamento de {model_name}...")
                # Verificar se há valores categóricos ou valores nulos antes do treinamento
                print(f"  - Tipos de dados nas features de treino:")
                # Usar dataset completo para treino inicial (não splitted) para debugging
                X_debug = self.df[self.train_features]
                for col in X_debug.columns:
                    print(f"    - {col}: {X_debug[col].dtype}, Nulos: {X_debug[col].isnull().sum()}")
                    
                grid_search.fit(X_debug, self.df[self.target])
                print(f"  - Treinamento concluído com sucesso!")
                
                # Avaliar no conjunto de validação
                val_score = roc_auc_score(
                    self.y_val, 
                    grid_search.predict_proba(self.X_val)[:, 1]
                )
                print(f"  - Melhor configuração: {grid_search.best_params_}")
                print(f"  - AUC-ROC Validação: {val_score:.4f}")
                
                # Atualizar melhor modelo se este for superior
                if val_score > best_score:
                    best_score = val_score
                    best_model = grid_search.best_estimator_
                    best_model_name = model_name
                    
            except Exception as e:
                print(f"⚠️ Erro ao treinar {model_name}: {str(e)}")
                print("  - Continuando com próximo modelo...")
        
        # Usar o melhor modelo encontrado
        print(f"\n✅ Melhor modelo: {best_model_name} (AUC-ROC: {best_score:.4f})")
        self.scoring_model = best_model
        
        # Extrair importâncias de features
        if hasattr(self.scoring_model, 'feature_importances_'):
            self.feature_importances = pd.DataFrame({
                'Feature': self.train_features,
                'Importance': self.scoring_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\n📊 Top 10 features mais importantes:")
            for idx, row in self.feature_importances.head(10).iterrows():
                print(f"  - {row['Feature']}: {row['Importance']:.4f}")
            
            # Salvar importância das features
            self.feature_importances.to_csv('data/insights/feature_importances.csv', index=False)
            
        # Avaliar no conjunto de teste
        y_pred_proba = self.scoring_model.predict_proba(self.X_test)[:, 1]
        y_pred = self.scoring_model.predict(self.X_test)
        
        test_auc = roc_auc_score(self.y_test, y_pred_proba)
        test_acc = accuracy_score(self.y_test, y_pred)
        
        print(f"\n📊 Performance final no conjunto de teste:")
        print(f"  - AUC-ROC: {test_auc:.4f}")
        print(f"  - Acurácia: {test_acc:.4f}")
        
        # Salvar modelo
        with open('models/scoring_model.pkl', 'wb') as f:
            pickle.dump(self.scoring_model, f)
            
        print("✅ Modelo de scoring salvo em models/scoring_model.pkl")
        return self.scoring_model
        
    def train_clustering_model(self):
        """
        Treina um modelo de clustering para agrupar candidatos e vagas similares
        Usa PCA para redução de dimensionalidade antes do clustering
        """
        print("\n🔄 Treinando modelo de clustering para agrupar perfis similares...")
        
        # Selecionar apenas features numéricas para clustering
        numeric_features = self.feature_groups['numeric']
        
        # Filtrar dados para remover possíveis NaN
        cluster_data = self.df[numeric_features].fillna(0)
        
        # Normalizar dados para clustering
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Redução de dimensionalidade com PCA
        pca = PCA(n_components=min(10, len(numeric_features)))
        reduced_data = pca.fit_transform(scaled_data)
        
        print(f"  - Redução de dimensionalidade: {cluster_data.shape[1]} -> {reduced_data.shape[1]} dimensões")
        print(f"  - Variância explicada: {sum(pca.explained_variance_ratio_):.2f}")
        
        # Encontrar número ideal de clusters usando o método do cotovelo
        inertias = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(reduced_data)
            inertias.append(kmeans.inertia_)
        
        # Usar o número de clusters definido na inicialização
        self.clustering_model = KMeans(
            n_clusters=self.n_clusters, 
            random_state=42,
            n_init=10
        )
        
        # Treinar modelo final
        self.clustering_model.fit(reduced_data)
        
        # Adicionar cluster aos dados
        self.df['cluster'] = self.clustering_model.predict(reduced_data)
        
        # Analisar distribuição de clusters
        cluster_dist = self.df['cluster'].value_counts(normalize=True) * 100
        print("\n📊 Distribuição de clusters:")
        for cluster, pct in cluster_dist.items():
            cluster_success_rate = self.df[self.df['cluster'] == cluster][self.target].mean() * 100
            print(f"  - Cluster {cluster}: {pct:.1f}% dos dados (Taxa de sucesso: {cluster_success_rate:.1f}%)")
        
        # Salvar componentes do modelo de clustering
        clustering_components = {
            'scaler': scaler,
            'pca': pca,
            'kmeans': self.clustering_model
        }
        
        with open('models/clustering_components.pkl', 'wb') as f:
            pickle.dump(clustering_components, f)
            
        print("✅ Modelo de clustering salvo em models/clustering_components.pkl")
        
        # Salvar insights sobre clusters
        self._analyze_clusters(reduced_data)
        
        return self.clustering_model
        
    def _analyze_clusters(self, reduced_data):
        """Analisa características de cada cluster para gerar insights"""
        cluster_insights = {}
        
        for cluster in range(self.n_clusters):
            # Filtrar dados deste cluster
            cluster_df = self.df[self.df['cluster'] == cluster]
            
            # Estatísticas básicas
            cluster_size = len(cluster_df)
            success_rate = cluster_df[self.target].mean() * 100
            
            # Características distintivas (baseadas em desvios em relação à média)
            distinctive_features = {}
            for feature in self.feature_groups['numeric']:
                if feature in self.df.columns:
                    # Calcular z-score da média do cluster em relação à média geral
                    cluster_mean = cluster_df[feature].mean()
                    overall_mean = self.df[feature].mean()
                    overall_std = self.df[feature].std()
                    
                    if overall_std > 0:  # Evitar divisão por zero
                        z_score = (cluster_mean - overall_mean) / overall_std
                        if abs(z_score) > 1.0:  # Mostrar apenas desvios significativos
                            distinctive_features[feature] = z_score
            
            # Guardar insights deste cluster
            cluster_insights[cluster] = {
                'size': cluster_size,
                'size_percent': (cluster_size / len(self.df)) * 100,
                'success_rate': success_rate,
                'distinctive_features': dict(sorted(distinctive_features.items(), 
                                               key=lambda x: abs(x[1]), 
                                               reverse=True)[:5])  # Top 5 características
            }
        
        # Salvar insights
        with open('data/insights/cluster_insights.pkl', 'wb') as f:
            pickle.dump(cluster_insights, f)
            
        # Visualizar clusters (usando primeiras 2 dimensões do PCA)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                           c=self.df['cluster'], cmap='viridis', 
                           alpha=0.6, s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.title('Visualização de Clusters de Candidatos/Vagas')
        plt.tight_layout()
        plt.savefig('data/visualizations/clustering_visualization.png')
        
        print("✅ Análise de clusters salva em data/insights/cluster_insights.pkl")
        print("✅ Visualização de clusters salva em data/visualizations/clustering_visualization.png")
        
        return cluster_insights
        
    def build_recommendation_system(self):
        """
        Constrói um sistema de recomendação combinando scoring e clustering
        para melhorar o match entre candidatos e vagas
        """
        print("\n🔄 Construindo sistema de recomendação híbrido...")
        
        # Verificar se temos modelos treinados
        if self.scoring_model is None or self.clustering_model is None:
            print("⚠️ É necessário treinar os modelos de scoring e clustering primeiro")
            return None
            
        # Criar feature que combina probabilidade de sucesso e cluster
        success_proba = self.scoring_model.predict_proba(self.df[self.train_features])[:, 1]
        self.df['success_probability'] = success_proba
        
        # Analisar probabilidade de sucesso por cluster
        print("\n📊 Probabilidade média de sucesso por cluster:")
        for cluster in range(self.n_clusters):
            cluster_proba = self.df[self.df['cluster'] == cluster]['success_probability'].mean()
            print(f"  - Cluster {cluster}: {cluster_proba:.4f}")
            
        # Criar score híbrido que combina probabilidade e informação de cluster
        # Este score pode ser usado para ranquear candidatos para uma vaga específica
        cluster_success_rate = self.df.groupby('cluster')[self.target].mean()
        
        # Mapear cada registro com score baseado em probabilidade individual e cluster
        self.df['hybrid_score'] = self.df.apply(
            lambda row: 0.7 * row['success_probability'] + 
                     0.3 * cluster_success_rate[row['cluster']], 
            axis=1
        )
        
        # Salvar modelo de recomendação (que inclui referências ao scoring e clustering)
        recommendation_system = {
            'scoring_model': self.scoring_model,
            'clustering_components': {
                'scaler': StandardScaler().fit(self.df[self.feature_groups['numeric']].fillna(0)),
                'pca': PCA(n_components=min(10, len(self.feature_groups['numeric']))).fit(
                    StandardScaler().fit_transform(self.df[self.feature_groups['numeric']].fillna(0))
                ),
                'kmeans': self.clustering_model
            },
            'cluster_success_rates': cluster_success_rate.to_dict(),
            'feature_groups': self.feature_groups,
            'train_features': self.train_features
        }
        
        with open('models/recommendation_system.pkl', 'wb') as f:
            pickle.dump(recommendation_system, f)
            
        print("✅ Sistema de recomendação salvo em models/recommendation_system.pkl")
        
        # Exportar dados com scores para análise
        self.df[['codigo', 'job_id', 'target_sucesso', 'cluster', 'success_probability', 'hybrid_score']].to_csv(
            'data/insights/recommendation_scores.csv', index=False
        )
        
        print("✅ Scores de recomendação salvos em data/insights/recommendation_scores.csv")
        return recommendation_system
        
    def run_pipeline(self):
        """Executa todo o pipeline de treinamento e avaliação"""
        print("\n" + "="*70)
        print("🚀 INICIANDO PIPELINE DE TREINAMENTO - SISTEMA HÍBRIDO SCORING + CLUSTERING")
        print("="*70)
        
        try:
            # Carregar e preparar dados
            self.df = self.load_data()
            self.train_features = self.prepare_features()
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.split_data()
            
            # Treinar modelo de scoring
            print("\n🔍 Verificando dados antes do treino:")
            print(f"  - Conjunto de treino: {type(self.X_train)}, shape: {self.X_train.shape if hasattr(self.X_train, 'shape') else 'N/A'}")
            print(f"  - Target treino: {type(self.y_train)}, valores únicos: {np.unique(self.y_train) if hasattr(self.y_train, 'unique') else 'N/A'}")
            
            # Treinar modelos
            self.scoring_model = self.train_scoring_model()
            self.clustering_model = self.train_clustering_model()
            
            # Construir sistema de recomendação
            self.build_recommendation_system()
            
            print("\n" + "="*70)
            print("✅ PIPELINE DE TREINAMENTO CONCLUÍDO COM SUCESSO")
            print("="*70)
            
            print("\nModelos salvos:")
            print("  - Modelo de Scoring: models/scoring_model.pkl")
            print("  - Modelo de Clustering: models/clustering_components.pkl")
            print("  - Sistema de Recomendação: models/recommendation_system.pkl")
            
            print("\nInsights gerados:")
            print("  - Importância de Features: data/insights/feature_importances.csv")
            print("  - Análise de Clusters: data/insights/cluster_insights.pkl")
            print("  - Scores de Recomendação: data/insights/recommendation_scores.csv")
            
            print("\nVisualizações:")
            print("  - Clusters: data/visualizations/clustering_visualization.png")
        
        except Exception as e:
            print(f"\n❌ ERRO DURANTE EXECUÇÃO DO PIPELINE: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Retornar a instância para facilitar uso programático
        return self


def main():
    """Função principal para executar o treinamento do modelo"""
    # Verificar se diretórios existem, caso contrário criá-los
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/insights', exist_ok=True)
    os.makedirs('data/visualizations', exist_ok=True)
    
    # Inicializar e executar o pipeline de treinamento
    trainer = ModelTrainer()
    trainer.run_pipeline()


if __name__ == "__main__":
    main()

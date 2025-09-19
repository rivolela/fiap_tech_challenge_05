# data_validation.py
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Tentar importar SMOTE, mas não falhar se não estiver disponível
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

class DataValidator:
    def __init__(self, features_path='data/processed/features_engineered.csv', 
                 target_path='data/processed/target_variable.csv',
                 complete_data_path='data/processed/complete_processed_data.csv'):
        """
        Inicializa o validador de dados com os caminhos para os arquivos processados
        
        Args:
            features_path: Caminho para o arquivo de features engenheiradas
            target_path: Caminho para o arquivo da variável target
            complete_data_path: Caminho para o arquivo com todos os dados processados
        """
        self.X = pd.read_csv(features_path)
        self.y = pd.read_csv(target_path)['target_sucesso']
        
        # Carregar dados completos para análises mais profundas
        try:
            self.complete_data = pd.read_csv(complete_data_path)
            print(f"✅ Dados carregados: {len(self.X)} amostras, {len(self.X.columns)} features")
        except Exception as e:
            print(f"⚠️ Erro ao carregar dados completos: {e}")
            self.complete_data = None
            
        # Carregar encoders e scalers para garantir consistência
        self.label_encoders = {}
        self.scaler = None
        
    def load_transformers(self):
        """Carrega transformadores salvos durante feature engineering"""
        models_dir = 'models'
        
        # Carregar label encoders
        try:
            if os.path.exists(os.path.join(models_dir, 'label_encoder.pkl')):
                with open(os.path.join(models_dir, 'label_encoder.pkl'), 'rb') as f:
                    self.label_encoders = pickle.load(f)
                print(f"✅ Label encoders carregados para {len(self.label_encoders)} features")
            else:
                self.label_encoders = {}
        except Exception as e:
            print(f"⚠️ Erro ao carregar label encoders: {e}")
            self.label_encoders = {}
            
        # Carregar scaler
        try:
            if os.path.exists(os.path.join(models_dir, 'feature_scaler.pkl')):
                with open(os.path.join(models_dir, 'feature_scaler.pkl'), 'rb') as f:
                    self.scaler = pickle.load(f)
                print("✅ Feature scaler carregado")
            else:
                self.scaler = None
        except Exception as e:
            print(f"⚠️ Erro ao carregar feature scaler: {e}")
            self.scaler = None
    
    def validate_data_quality(self):
        """Valida qualidade dos dados com base nas insights da análise exploratória"""
        print("="*50)
        print("VALIDAÇÃO DA QUALIDADE DOS DADOS")
        print("="*50)
        
        # Verificar valores faltantes
        missing_data = self.X.isnull().sum()
        missing_percentage = (missing_data / len(self.X)) * 100
        
        print(f"\n📊 VALORES FALTANTES:")
        missing_cols = missing_data[missing_data > 0]
        if len(missing_cols) > 0:
            for col in missing_cols.index:
                print(f"  {col}: {missing_data[col]} ({missing_percentage[col]:.1f}%)")
        else:
            print("  ✅ Nenhum valor faltante encontrado!")
            
        # Verificar duplicatas
        duplicates = self.X.duplicated().sum()
        print(f"\n🔄 DUPLICATAS: {duplicates} registros duplicados")
        
        # Verificar outliers usando IQR
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns
        print(f"\n📈 OUTLIERS DETECTADOS:")
        
        for col in numeric_cols:
            Q1 = self.X[col].quantile(0.25)
            Q3 = self.X[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.X[col] < (Q1 - 1.5 * IQR)) | (self.X[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                print(f"  {col}: {outliers} outliers ({(outliers/len(self.X))*100:.1f}%)")
                
        # Verificar balanceamento do target
        target_dist = self.y.value_counts()
        print(f"\n🎯 DISTRIBUIÇÃO DO TARGET:")
        for value, count in target_dist.items():
            print(f"  Classe {value}: {count} ({(count/len(self.y))*100:.1f}%)")
        
        # Calcular desequilíbrio de classes
        if len(target_dist) > 1:
            minority_class = target_dist.min()
            majority_class = target_dist.max()
            imbalance_ratio = majority_class / minority_class
            print(f"\n⚖️ DESEQUILÍBRIO DE CLASSES: 1:{imbalance_ratio:.1f}")
            
            if imbalance_ratio > 10:
                print("  ⚠️ AVISO: Desequilíbrio extremo de classes. Recomenda-se usar técnicas de balanceamento.")
                
        # Análise das features mais importantes identificadas no feature engineering
        if 'engajamento_positivo' in self.X.columns:
            pos_engagement_rate = self.X['engajamento_positivo'].mean() * 100
            print(f"\n📈 ESTATÍSTICAS IMPORTANTES:")
            print(f"  Taxa de engajamento positivo: {pos_engagement_rate:.1f}%")
            
        # Verificar variância das features
        low_var_cols = []
        for col in self.X.columns:
            if self.X[col].nunique() <= 5:  # Para features categóricas/binárias
                counts = self.X[col].value_counts(normalize=True)
                if counts.max() > 0.95:  # Se mais de 95% dos valores são iguais
                    low_var_cols.append((col, counts.max() * 100))
                    
        if low_var_cols:
            print(f"\n⚠️ FEATURES COM BAIXA VARIÂNCIA:")
            for col, pct in sorted(low_var_cols, key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {col}: {pct:.1f}% dos valores concentrados em uma única categoria")
            
    def balance_training_data(self, X_train, y_train):
        """Balanceia os dados de treinamento para lidar com desbalanceamento de classes"""
        # Usar SMOTE para balancear se disponível, caso contrário usar random oversampling
        if SMOTE_AVAILABLE:
            try:
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                print(f"✅ Dados de treino balanceados usando SMOTE")
                return X_resampled, y_resampled
            except Exception as e:
                print(f"⚠️ Erro ao aplicar SMOTE: {e}")
                print("⚠️ Usando random oversampling como alternativa")
                return self._apply_random_oversampling(X_train, y_train)
        else:
            print("⚠️ SMOTE não disponível, usando resampling simples como alternativa")
            return self._apply_random_oversampling(X_train, y_train)
            
    def _apply_random_oversampling(self, X_train, y_train):
        """Implementação simples de random oversampling para balancear classes"""
        # Identificar a classe minoritária e majoritária
        class_counts = y_train.value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        
        # Selecionar exemplos da classe minoritária
        minority_indices = y_train[y_train == minority_class].index
        majority_indices = y_train[y_train == majority_class].index
        
        # Repetir exemplos da classe minoritária até balancear
        minority_resampled = np.random.choice(
            minority_indices, 
            size=len(majority_indices), 
            replace=True
        )
        
        # Combinar com exemplos da classe majoritária
        balanced_indices = np.concatenate([majority_indices, minority_resampled])
        
        # Criar conjuntos de dados balanceados
        X_resampled = X_train.loc[balanced_indices].reset_index(drop=True)
        y_resampled = y_train.loc[balanced_indices].reset_index(drop=True)
        
        print(f"✅ Dados de treino balanceados usando random oversampling")
        return X_resampled, y_resampled

    def create_train_validation_split(self, test_size=0.2, val_size=0.25, random_state=42):
        """
        Cria divisões estratificadas para treino, validação e teste com opção de balanceamento
        
        Args:
            test_size: Porcentagem dos dados para o conjunto de teste
            val_size: Porcentagem dos dados restantes para validação
            random_state: Semente para reprodutibilidade
        """
        # Primeira divisão: 80% treino+validação, 20% teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Segunda divisão: 75% treino, 25% validação (do conjunto treino+validação)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        print(f"\n✅ DIVISÃO DOS DADOS:")
        print(f"  Treino: {X_train.shape[0]} amostras ({(X_train.shape[0]/len(self.X))*100:.1f}%)")
        print(f"  Validação: {X_val.shape[0]} amostras ({(X_val.shape[0]/len(self.X))*100:.1f}%)")
        print(f"  Teste: {X_test.shape[0]} amostras ({(X_test.shape[0]/len(self.X))*100:.1f}%)")
        
        # Verificar desequilíbrio no conjunto de treino
        train_class_dist = y_train.value_counts()
        if len(train_class_dist) > 1:
            minority_class = train_class_dist.min()
            majority_class = train_class_dist.max()
            imbalance_ratio = majority_class / minority_class
            
            if imbalance_ratio > 5:
                print(f"\n⚠️ ALTA DESPROPORÇÃO NAS CLASSES: 1:{imbalance_ratio:.1f}")
                # Aplicar balanceamento
                X_train, y_train = self.balance_training_data(X_train, y_train)
        
        # Salvar splits
        splits_dir = 'data/processed/splits'
        if not os.path.exists(splits_dir):
            os.makedirs(splits_dir)
            
        # Criar diretório para visualizações
        viz_dir = 'data/visualizations'
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        datasets = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        for split_name, (X_split, y_split) in datasets.items():
            X_split.to_csv(f'{splits_dir}/X_{split_name}.csv', index=False)
            y_split.to_csv(f'{splits_dir}/y_{split_name}.csv', index=False)
            
        print(f"\n✅ Conjuntos de dados salvos em {splits_dir}/")
        return datasets
        
    def generate_feature_analysis(self):
        """
        Gera análise detalhada das features com base nos insights da análise exploratória
        Identifica features mais importantes para prever sucesso na contratação
        """
        print("\n" + "="*50)
        print("📊 ANÁLISE DE FEATURES PARA PREVISÃO DE SUCESSO")
        print("="*50)
        
        # 1. Correlação com o target
        correlations = []
        for col in self.X.columns:
            if self.X[col].dtype in ['int64', 'float64']:
                try:
                    corr = self.X[col].corr(self.y)
                    correlations.append({'feature': col, 'correlation': corr})
                except Exception as e:
                    print(f"⚠️ Erro ao calcular correlação para {col}: {e}")
                
        corr_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
        
        print(f"\n📊 TOP 10 FEATURES POR CORRELAÇÃO COM TARGET:")
        for _, row in corr_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['correlation']:.3f}")
        
        # 2. Agrupar features por categorias com base no feature engineering
        feature_categories = {
            'Engajamento': [col for col in self.X.columns if 'engajamento' in col],
            'Recrutador': [col for col in self.X.columns if 'recrutador' in col],
            'Comentários': [col for col in self.X.columns if 'menciona_' in col or 'sentimento' in col],
            'Temporais': [col for col in self.X.columns if any(term in col for term in ['data', 'dia', 'mes', 'tempo'])],
            'Vaga': [col for col in self.X.columns if any(term in col for term in ['vaga', 'area', 'senioridade', 'contratacao'])]
        }
        
        print("\n📑 FEATURES POR CATEGORIA:")
        for category, features in feature_categories.items():
            if features:
                avg_corr = np.mean([abs(corr_df[corr_df['feature'] == f]['correlation'].values[0]) 
                                  for f in features if f in corr_df['feature'].values])
                print(f"  {category}: {len(features)} features (correlação média: {avg_corr:.3f})")
                
        # 3. Análise específica de features de engajamento
        engagement_features = [col for col in self.X.columns if any(term in col for term in 
                                                                ['engajamento', 'sentimento', 'menciona_'])]
        # Criar um DataFrame temporário combinando features e target para análise
        temp_df = self.X.copy()
        temp_df['target'] = self.y.values
        
        if engagement_features:
            print("\n🔍 ANÁLISE DE FEATURES DE ENGAJAMENTO:")
            for feature in engagement_features[:5]:  # Mostrar os 5 primeiros
                if feature in corr_df['feature'].values:
                    corr_val = corr_df[corr_df['feature'] == feature]['correlation'].values[0]
                    positive_success_rate = temp_df[temp_df[feature] == 1]['target'].mean() * 100
                    negative_success_rate = temp_df[temp_df[feature] == 0]['target'].mean() * 100
                    print(f"  {feature}:")
                    print(f"    - Correlação com sucesso: {corr_val:.3f}")
                    print(f"    - Taxa de sucesso quando positivo: {positive_success_rate:.1f}%")
                    print(f"    - Taxa de sucesso quando negativo: {negative_success_rate:.1f}%")
        
        # 4. Features relacionadas à desistência (um dos principais insights da análise)
        if 'menciona_desistencia' in self.X.columns or 'menciona_outra_proposta' in self.X.columns:
            print("\n⚠️ ANÁLISE DE FATORES DE DESISTÊNCIA:")
            
            desistencia_features = [col for col in self.X.columns if any(term in col for term in 
                                                                    ['desistencia', 'proposta', 'recusou'])]
            
            for feature in desistencia_features:
                if feature in self.X.columns:
                    success_rate = temp_df.groupby(feature)['target'].mean()
                    print(f"  {feature}:")
                    print(f"    - Taxa de sucesso quando presente: {success_rate.get(1, 0)*100:.1f}%")
                    print(f"    - Taxa de sucesso quando ausente: {success_rate.get(0, 0)*100:.1f}%")
        
        # Salvar análises
        if not os.path.exists('data/insights'):
            os.makedirs('data/insights')
            
        corr_df.to_csv('data/insights/feature_correlations.csv', index=False)
        
        # Salvar importância das categorias
        category_importance = {}
        for category, features in feature_categories.items():
            if features:
                valid_features = [f for f in features if f in corr_df['feature'].values]
                if valid_features:
                    avg_corr = np.mean([abs(corr_df[corr_df['feature'] == f]['correlation'].values[0]) 
                                      for f in valid_features])
                    category_importance[category] = avg_corr
        
        pd.DataFrame([{'categoria': k, 'importancia_media': v} 
                     for k, v in category_importance.items()]).to_csv(
            'data/insights/category_importance.csv', index=False)
            
        print(f"\n✅ Análises salvas em data/insights/")
        return corr_df

    def validate_target_patterns(self):
        """
        Valida padrões específicos da variável target com base nos insights da análise exploratória
        """
        print("\n" + "="*50)
        print("🎯 VALIDAÇÃO DE PADRÕES DA VARIÁVEL TARGET")
        print("="*50)
        
        # Se tivermos dados completos, podemos fazer análises mais detalhadas
        if self.complete_data is not None and 'categoria_situacao' in self.complete_data.columns:
            # Analisar sucesso por categoria de situação
            situation_counts = self.complete_data['categoria_situacao'].value_counts()
            print(f"\n📊 DISTRIBUIÇÃO POR CATEGORIA:")
            for cat, count in situation_counts.items():
                print(f"  {cat}: {count} ({count/len(self.complete_data)*100:.1f}%)")
            
            # Situações originais mapeadas para cada categoria
            if 'situacao_candidato' in self.complete_data.columns:
                print("\n🔄 SITUAÇÕES ORIGINAIS POR CATEGORIA:")
                for cat in ['sucesso', 'fracasso', 'em_andamento', 'indefinido']:
                    if cat in situation_counts.index:
                        subset = self.complete_data[self.complete_data['categoria_situacao'] == cat]
                        print(f"\n  {cat.upper()}:")
                        situation_dist = subset['situacao_candidato'].value_counts().head(5)
                        for situation, count in situation_dist.items():
                            print(f"    - {situation}: {count} ({count/len(subset)*100:.1f}%)")
                            
        # Analisar features relevantes em relação ao target
        print("\n🔍 VALIDAÇÃO DE FEATURES IMPORTANTES:")
        key_features = [
            'engajamento_positivo', 'menciona_desistencia', 'menciona_outra_proposta', 
            'fit_cultural', 'taxa_sucesso_recrutador', 'recrutador_alto_desempenho'
        ]
        
        # Criar um DataFrame temporário combinando features e target para análise
        temp_df = self.X.copy()
        temp_df['target'] = self.y.values
        
        for feature in key_features:
            if feature in self.X.columns:
                success_rate = temp_df.groupby(feature)['target'].mean() * 100
                print(f"\n  {feature}:")
                if 1 in success_rate.index and 0 in success_rate.index:
                    print(f"    - Taxa de sucesso quando positivo: {success_rate[1]:.1f}%")
                    print(f"    - Taxa de sucesso quando negativo: {success_rate[0]:.1f}%")
                    
                    # Calcular o lift (quanto a presença da feature multiplica a chance de sucesso)
                    baseline_rate = self.y.mean() * 100
                    lift = success_rate[1] / baseline_rate
                    print(f"    - Lift: {lift:.2f}x a taxa média")
        
        return situation_counts if self.complete_data is not None else None

    def evaluate_simple_model(self, cv_folds=5):
        """
        Avalia rapidamente um modelo simples para verificar o poder preditivo das features
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import classification_report, roc_auc_score
            
            print("\n" + "="*50)
            print("🤖 AVALIAÇÃO RÁPIDA DO PODER PREDITIVO")
            print("="*50)
            
            # Separar dados para validação cruzada
            kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Modelo simples para verificar poder preditivo
            model = LogisticRegression(max_iter=1000, class_weight='balanced')
            
            # Validação cruzada
            cv_scores = cross_val_score(model, self.X, self.y, cv=kfold, scoring='roc_auc')
            
            print(f"\n📊 PERFORMANCE DO MODELO LOGISTIC REGRESSION:")
            print(f"  AUC-ROC médio (validação cruzada): {cv_scores.mean():.3f}")
            print(f"  Desvio padrão: {cv_scores.std():.3f}")
            
            # Avaliar eficácia relativa às categorias de features
            feature_categories = {
                'Engajamento': [col for col in self.X.columns if any(term in col for term in 
                                                                ['engajamento', 'fit', 'interesse'])],
                'Recrutador': [col for col in self.X.columns if 'recrutador' in col],
                'Texto': [col for col in self.X.columns if any(term in col for term in 
                                                            ['menciona_', 'sentimento', 'comentario'])]
            }
            
            print("\n📈 AVALIAÇÃO POR CATEGORIA DE FEATURES:")
            for category, features in feature_categories.items():
                valid_features = [f for f in features if f in self.X.columns]
                if valid_features:
                    X_subset = self.X[valid_features]
                    cv_scores = cross_val_score(model, X_subset, self.y, cv=kfold, scoring='roc_auc')
                    print(f"  {category}: AUC-ROC = {cv_scores.mean():.3f}")
                    
            print("\n✅ Avaliação de modelo concluída!")
            return cv_scores.mean()
            
        except Exception as e:
            print(f"⚠️ Erro ao avaliar modelo: {e}")
            return None

def run_validation():
    """Executa todo o pipeline de validação"""
    print("\n" + "="*50)
    print("🚀 INICIANDO VALIDAÇÃO DOS DADOS")
    print("="*50 + "\n")
    
    validator = DataValidator()
    validator.validate_data_quality()
    validator.validate_target_patterns()
    feature_analysis = validator.generate_feature_analysis()
    datasets = validator.create_train_validation_split()
    model_score = validator.evaluate_simple_model()
    
    print("\n" + "="*50)
    print("✅ VALIDAÇÃO DOS DADOS CONCLUÍDA!")
    print(f"✅ POWER SCORE: {model_score:.4f}")
    print("✅ Dados prontos para treinamento dos modelos")
    print("="*50)
    
# Executar validação
if __name__ == "__main__":
    run_validation()

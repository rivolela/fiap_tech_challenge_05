# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
import pickle
import os
from datetime import datetime, timedelta
import nltk
from textblob import TextBlob

class DecisionFeatureEngineer:
    def __init__(self, data_path='data/processed/prospects_processed.csv', 
                jobs_path='data/processed/jobs_processed.csv',
                applicants_path='data/processed/applicants_processed.csv'):
        self.df = pd.read_csv(data_path)
        self.label_encoders = {}
        self.vectorizers = {}
        self.scaler = StandardScaler()
        
        # Carregar dados complementares se existirem
        try:
            self.jobs_df = pd.read_csv(jobs_path) if os.path.exists(jobs_path) else None
            self.applicants_df = pd.read_csv(applicants_path) if os.path.exists(applicants_path) else None
            
            # Mesclar dados de jobs e applicants se disponíveis
            if self.jobs_df is not None:
                print(f"✅ Dados de vagas carregados: {len(self.jobs_df)} registros")
                self.df = self.df.merge(self.jobs_df, on='job_id', how='left')
                
            if self.applicants_df is not None:
                print(f"✅ Dados de candidatos carregados: {len(self.applicants_df)} registros")
                if 'applicant_id' in self.df.columns and 'applicant_id' in self.applicants_df.columns:
                    self.df = self.df.merge(self.applicants_df, on='applicant_id', how='left')
        except Exception as e:
            print(f"⚠️ Erro ao carregar dados complementares: {e}")
            
        # Mapeamento de situações conforme data_analysis
        self.situacoes_map = {
            'sucesso': ['Contratado pela Decision', 'Contratado como Hunting', 'Aprovado',
                       'Proposta Aceita', 'Encaminhar Proposta', 'Documentação PJ', 
                       'Documentação CLT', 'Documentação Cooperado'],
            'em_andamento': ['Prospect', 'Inscrito', 'Encaminhado ao Requisitante', 
                           'Entrevista Técnica', 'Entrevista com Cliente', 'Em avaliação pelo RH'],
            'fracasso': ['No Aprovado pelo Cliente', 'Não Aprovado pelo Cliente', 'No Aprovado pelo RH',
                        'Não Aprovado pelo RH', 'No Aprovado pelo Requisitante', 'Não Aprovado pelo Requisitante',
                        'Desistiu', 'Desistiu da Contratação', 'Sem interesse nesta vaga', 'Recusado']
        }
        
    def extract_temporal_features(self):
        """Extrai features temporais baseadas na análise"""
        # Identificar colunas de data (mais flexível para lidar com diferentes nomes)
        date_cols = {
            'data_candidatura': ['datacandidatura', 'data_candidatura', 'data'],
            'data_atualizacao': ['ultimaatualizacao', 'ultima_atualizacao', 'data_atualizacao']
        }
        
        # Encontrar as colunas corretas
        for target_col, possible_cols in date_cols.items():
            for col in possible_cols:
                if col in self.df.columns:
                    self.df[target_col] = pd.to_datetime(self.df[col], errors='coerce')
                    break
                    
        # Verificar se conseguimos identificar as colunas de data
        if 'data_candidatura' not in self.df.columns:
            print("⚠️ Não foi possível identificar a coluna de data de candidatura")
            # Criar uma data fictícia para não quebrar o código
            self.df['data_candidatura'] = pd.Timestamp('2021-01-01')
            
        if 'data_atualizacao' not in self.df.columns:
            # Se não tiver data de atualização, usar a data de candidatura
            if 'data_candidatura' in self.df.columns:
                self.df['data_atualizacao'] = self.df['data_candidatura']
            else:
                self.df['data_atualizacao'] = pd.Timestamp('2021-01-01')
        
        # Calcular tempo no pipeline (importante para prever o desfecho)
        self.df['dias_no_pipeline'] = (self.df['data_atualizacao'] - self.df['data_candidatura']).dt.days
        self.df['dias_no_pipeline'] = self.df['dias_no_pipeline'].fillna(0).clip(0, 365)  # Limitar outliers
        
        # Features sazonais
        self.df['mes_candidatura'] = self.df['data_candidatura'].dt.month
        self.df['ano_candidatura'] = self.df['data_candidatura'].dt.year
        self.df['dia_semana_candidatura'] = self.df['data_candidatura'].dt.dayofweek
        self.df['trimestre_candidatura'] = self.df['data_candidatura'].dt.quarter
        self.df['fim_do_mes'] = (self.df['data_candidatura'].dt.day > 25).astype(int)
        self.df['inicio_do_mes'] = (self.df['data_candidatura'].dt.day <= 5).astype(int)
        
        print("✅ Features temporais extraídas")
        
    def extract_text_features(self):
        """Extrai features dos comentários com base na análise detalhada de data_analysis"""
        def analyze_comment_sentiment(comment):
            if pd.isna(comment):
                return 0
            try:
                blob = TextBlob(str(comment))
                return blob.sentiment.polarity
            except:
                # Se der erro, usar uma abordagem mais simples baseada em palavras-chave
                palavras_positivas = ['aprovado', 'contratado', 'encaminhado', 'interessado', 'motivado', 
                                      'entusiasmado', 'ótimo', 'excelente', 'positivo']
                palavras_negativas = ['não', 'reprovado', 'desistiu', 'declinou', 'rejeitado', 
                                     'sem interesse', 'preocupado', 'insatisfeito']
                                     
                comment_lower = str(comment).lower()
                sentimento = 0
                
                for palavra in palavras_positivas:
                    if palavra in comment_lower:
                        sentimento += 1
                        
                for palavra in palavras_negativas:
                    if palavra in comment_lower:
                        sentimento -= 1
                        
                return max(min(sentimento / 5, 1), -1)  # Normalizar entre -1 e 1
            
        def extract_salary_from_comment(comment):
            if pd.isna(comment):
                return None
            # Regex mais robusto para capturar salários
            patterns = [
                r'R\$?\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)',
                r'(\d{1,3}(?:\.\d{3})*)\s*reais?',
                r'pretenso.*?(\d{1,3}(?:\.\d{3})*)',
                r'sal[aá]rio.*?(\d{1,3}(?:\.\d{3})*)',
                r'(\d{1,3}(?:\.\d{3})*)[kK]'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, str(comment), re.IGNORECASE)
                if match:
                    salary_str = match.group(1).replace('.', '').replace(',', '.')
                    try:
                        salary = float(salary_str)
                        # Se o valor for muito baixo, provavelmente está em milhares
                        if salary < 100 and 'k' in str(comment).lower():
                            salary *= 1000
                        return salary
                    except ValueError:
                        continue
            return None
            
        def extract_availability_days(comment):
            if pd.isna(comment):
                return None
            match = re.search(r'(\d+)\s*dias?', str(comment).lower())
            return int(match.group(1)) if match else None
            
        def detect_engagement(comment):
            """Detecta engajamento do candidato com verificação contextual avançada"""
            if pd.isna(comment):
                return False
                
            comment_lower = str(comment).lower()
            
            # Detectar presença de palavras de engajamento
            palavras_engajamento = [
                'interesse', 'interessado', 'motivado', 'disponível', 'empolgado', 'comprometido', 
                'dedicado', 'gostaria', 'quer', 'aceita', 'aceitou', 'topou', 'contente', 
                'animado', 'positivo', 'encaminhar', 'prosseguir', 'continuar', 'gostou', 
                'aprovou', 'contratado', 'contratação'
            ]
            
            # Detectar negações ou contexto negativo
            palavras_negativas_contexto = [
                'não tem', 'sem', 'não possui', 'não demonstrou', 'pouco', 'falta de', 
                'desistiu', 'declinou', 'recusou', 'não aceitou', 'não quer', 'não gostou',
                'não vai', 'não está', 'não pode', 'fora do'
            ]
            
            # Para candidatos contratados, considerar como engajamento positivo
            if 'contratado' in comment_lower or 'contratação' in comment_lower:
                # Verificar se não há negação direta
                if not any(neg in comment_lower for neg in ['não contratado', 'não foi contratado', 'desistiu da contratação']):
                    return True
            
            # Verificar engajamento no contexto
            tem_palavras_engajamento = any(palavra in comment_lower for palavra in palavras_engajamento)
            tem_contexto_negativo = any(neg in comment_lower for neg in palavras_negativas_contexto)
            
            return tem_palavras_engajamento and not tem_contexto_negativo
            
        def detect_fit_cultural(comment):
            """Detecta menção a fit cultural"""
            if pd.isna(comment):
                return False
                
            palavras_fit_cultural = [
                'fit', 'cultura', 'valores', 'adequado', 'perfil', 'alinhado', 'equipe', 
                'time', 'comportamental', 'compatível', 'adaptação'
            ]
            
            return any(palavra in str(comment).lower() for palavra in palavras_fit_cultural)
            
        def extract_technical_skills(comment):
            """Extrai habilidades técnicas mencionadas no comentário"""
            if pd.isna(comment):
                return []
                
            # Lista ampliada de habilidades técnicas com base na análise de dados
            habilidades = [
                'java', 'python', 'javascript', 'react', 'node', 'angular', 'aws', 'cloud', 
                'devops', 'scrum', 'agile', 'sap', 'abap', 'sql', 'database', 'front-end', 
                'backend', 'mobile', 'api', 'rest', 'spring', 'django', 'azure', 'gcp', 'microservices',
                'kubernetes', 'docker', 'ci/cd', 'git', 'arquitetura', 'performance'
            ]
            
            return [skill for skill in habilidades if re.search(r'\b' + skill + r'\b', str(comment).lower())]
            
        # Aplicar extração de features
        self.df['sentimento_comentario'] = self.df['comentario'].apply(analyze_comment_sentiment)
        self.df['pretensao_salarial_extraida'] = self.df['comentario'].apply(extract_salary_from_comment)
        self.df['disponibilidade_dias'] = self.df['comentario'].apply(extract_availability_days)
        self.df['engajamento_positivo'] = self.df['comentario'].apply(detect_engagement)
        self.df['fit_cultural'] = self.df['comentario'].apply(detect_fit_cultural)
        
        # Extrair habilidades técnicas e criar features para as principais
        self.df['habilidades_tecnicas'] = self.df['comentario'].apply(extract_technical_skills)
        for skill in ['java', 'python', 'sap', 'cloud', 'agile']:
            self.df[f'menciona_{skill}'] = self.df['habilidades_tecnicas'].apply(lambda x: skill in x)
        
        # Features de idioma
        self.df['menciona_ingles'] = self.df['comentario'].str.contains(r'ingl[eê]s', case=False, na=False)
        
        # Features adicionais baseadas nos achados da análise
        self.df['menciona_contratacao'] = self.df['comentario'].str.contains('contrata|pj|clt|cooperado', case=False, na=False)
        self.df['menciona_projeto'] = self.df['comentario'].str.contains('projeto|cliente', case=False, na=False)
        self.df['menciona_valores'] = self.df['comentario'].str.contains(r'R\$|valor|salário|remuneração', case=False, na=False)
        self.df['menciona_desistencia'] = self.df['comentario'].str.contains('desistiu|declinou|recusou|não aceitou', case=False, na=False)
        self.df['menciona_outra_proposta'] = self.df['comentario'].str.contains('outra proposta|outra oferta|outro processo', case=False, na=False)
        
        # Comprimento do comentário
        self.df['tamanho_comentario'] = self.df['comentario'].str.len().fillna(0)
        
        print("✅ Features de texto extraídas")
        
    def extract_categorical_features(self):
        """Processa features categóricas com base na análise de dados"""
        # Extrair senioridade da vaga
        def extract_seniority(job_title):
            if pd.isna(job_title):
                return 'Não Especificado'
            title_lower = str(job_title).lower()
            if any(term in title_lower for term in ['senior', 'sr', 'sênior', 'especialista', 'specialist']):
                return 'Senior'
            elif any(term in title_lower for term in ['junior', 'jr', 'júnior', 'trainee']):
                return 'Junior'
            elif any(term in title_lower for term in ['pleno', 'pl', 'mid']):
                return 'Pleno'
            else:
                return 'Não Especificado'
                
        # Extrair área da vaga baseado nos insights da análise (SAP, Desenvolvimento, Data/BI, etc.)
        def extract_job_area(job_title):
            if pd.isna(job_title):
                return 'Não Especificado'
                
            job_title = str(job_title).lower()
            
            if any(term in job_title for term in ['sap', 'abap', 'fico', 'mm', 'sd']):
                return 'SAP'
            elif any(term in job_title for term in ['java', 'python', '.net', 'dev', 'programador', 'desenvolvedor']):
                return 'Desenvolvimento'
            elif any(term in job_title for term in ['data', 'bi', 'analytics', 'big data', 'dados']):
                return 'Data/BI'
            elif any(term in job_title for term in ['devops', 'cloud', 'aws', 'infra']):
                return 'DevOps/Cloud'
            elif any(term in job_title for term in ['dba', 'oracle', 'database', 'banco de dados']):
                return 'Database'
            else:
                return 'Outros'
                
        # Extrair tipo de contratação
        def extract_contract_type(row):
            if 'informacoes_basicas' in row and isinstance(row['informacoes_basicas'], dict):
                return row['informacoes_basicas'].get('tipo_contratacao', 'Não Especificado')
            elif 'tipo_contratacao' in row:
                return row['tipo_contratacao']
            else:
                return 'Não Especificado'
        
        # Aplicar as extrações se as colunas existirem
        if 'titulo_vaga' in self.df.columns:
            self.df['senioridade_vaga'] = self.df['titulo_vaga'].apply(extract_seniority)
            self.df['area_vaga'] = self.df['titulo_vaga'].apply(extract_job_area)
        
        # Tentar extrair tipo de contratação
        try:
            self.df['tipo_contratacao'] = self.df.apply(extract_contract_type, axis=1)
        except:
            print("⚠️ Não foi possível extrair o tipo de contratação")
        
        # Processar categoria de situação para casos onde não foi definida
        if 'categoria_situacao' in self.df.columns:
            # Se já existir, não sobrescrever
            print(f"✅ Coluna categoria_situacao já existe com valores: {self.df['categoria_situacao'].unique()}")
        else:
            # Verificar qual é a coluna de situação
            situation_col = None
            for col in ['situacao_candidato', 'situacao_candidado', 'situacao', 'status']:
                if col in self.df.columns:
                    situation_col = col
                    break
                    
            if situation_col:
                # Classificar situações usando o mapeamento da análise de dados
                def classify_situation(situacao):
                    if pd.isna(situacao):
                        return 'indefinido'
                        
                    for categoria, situacoes in self.situacoes_map.items():
                        if situacao in situacoes:
                            return categoria
                    return 'indefinido'
                    
                self.df['categoria_situacao'] = self.df[situation_col].apply(classify_situation)
                
                # Target binário para sucesso (para modelos de classificação binária)
                self.df['target_sucesso'] = (self.df['categoria_situacao'] == 'sucesso').astype(int)
                
                print(f"✅ Variáveis target criadas: categoria_situacao e target_sucesso")
        
        # Tratar casos indefinidos com base na análise de texto
        if 'categoria_situacao' in self.df.columns and 'indefinido' in self.df['categoria_situacao'].values:
            # Tentar reclassificar casos indefinidos com base nos comentários
            indefinidos = self.df[self.df['categoria_situacao'] == 'indefinido']
            print(f"⚠️ Existem {len(indefinidos)} casos indefinidos, tentando reclassificar...")
            
            # Regras derivadas da análise de dados
            for idx in indefinidos.index:
                comment = str(self.df.loc[idx, 'comentario']).lower() if 'comentario' in self.df.columns else ''
                
                # Regras para identificar sucessos
                if any(term in comment for term in ['contratado', 'aprovado', 'aceito', 'proposta aceita']):
                    if not any(neg_term in comment for neg_term in ['não foi contratado', 'não aprovado']):
                        self.df.loc[idx, 'categoria_situacao'] = 'sucesso'
                        self.df.loc[idx, 'target_sucesso'] = 1
                        continue
                
                # Regras para identificar fracassos
                if any(term in comment for term in ['desistiu', 'recusou', 'declinou', 'sem interesse', 'não aprovado']):
                    self.df.loc[idx, 'categoria_situacao'] = 'fracasso'
                    continue
                    
                # Regras para identificar em andamento
                if any(term in comment for term in ['em análise', 'avaliação', 'aguardando', 'entrevista marcada']):
                    self.df.loc[idx, 'categoria_situacao'] = 'em_andamento'
                    continue
            
            print(f"✅ Após reclassificação: {self.df['categoria_situacao'].value_counts().to_dict()}")
        
        # Encode categorical variables
        categorical_cols = ['recrutador', 'area_vaga', 'senioridade_vaga', 'modalidade', 'tipo_contratacao']
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].fillna('Não Especificado'))
                self.label_encoders[col] = le
                
                # Para colunas importantes, criar também one-hot encoding
                if col in ['area_vaga', 'senioridade_vaga', 'tipo_contratacao']:
                    # Limitar a colunas com poucos valores únicos para evitar explosão de dimensionalidade
                    if self.df[col].nunique() <= 10:
                        dummies = pd.get_dummies(self.df[col], prefix=col)
                        self.df = pd.concat([self.df, dummies], axis=1)
                
        print("✅ Features categóricas processadas")
        
    def create_aggregated_features(self):
        """Cria features agregadas baseadas nos insights da análise de dados"""
        # Taxa de sucesso por recrutador (identificado como importante na análise)
        if 'recrutador' in self.df.columns and 'target_sucesso' in self.df.columns:
            recruiter_success = self.df.groupby('recrutador')['target_sucesso'].agg(['mean', 'count']).reset_index()
            recruiter_success.columns = ['recrutador', 'taxa_sucesso_recrutador', 'total_candidatos_recrutador']
            self.df = self.df.merge(recruiter_success, on='recrutador', how='left')
            
            # Identificar recrutadores de alto desempenho
            self.df['recrutador_alto_desempenho'] = (self.df['taxa_sucesso_recrutador'] > 
                                                    self.df['taxa_sucesso_recrutador'].mean()).astype(int)
        
        # Features por área de vaga
        if 'area_vaga' in self.df.columns:
            # Popularidade da área de vaga
            area_counts = self.df['area_vaga'].value_counts().to_dict()
            self.df['popularidade_area'] = self.df['area_vaga'].map(area_counts)
            
            # Taxa de sucesso por área de vaga
            area_success = self.df.groupby('area_vaga')['target_sucesso'].mean().to_dict()
            self.df['taxa_sucesso_area'] = self.df['area_vaga'].map(area_success)
            
            # Tempo médio no pipeline por área
            if 'dias_no_pipeline' in self.df.columns:
                area_avg_time = self.df.groupby('area_vaga')['dias_no_pipeline'].mean().to_dict()
                self.df['tempo_medio_pipeline_area'] = self.df['area_vaga'].map(area_avg_time)
                
                # Comparação com o tempo médio geral (acima/abaixo)
                media_tempo_pipeline = self.df['dias_no_pipeline'].mean()
                self.df['tempo_acima_media'] = (self.df['dias_no_pipeline'] > media_tempo_pipeline).astype(int)
        
        # Features por tipo de contratação (identificado como fator de desistência)
        if 'tipo_contratacao' in self.df.columns:
            # Taxa de sucesso por tipo de contratação
            contract_success = self.df.groupby('tipo_contratacao')['target_sucesso'].mean().to_dict()
            self.df['taxa_sucesso_contratacao'] = self.df['tipo_contratacao'].map(contract_success)
            
            # Ranking de preferência de contratação
            # Os dados mostraram que PJ tem taxas diferentes de sucesso vs. CLT
            contract_types = {
                'PJ/Autônomo': 1,
                'CLT Full': 2,
                'Cooperado': 3,
                'Hunting': 4,
                'CLT Cotas': 5
            }
            
            # Criar um mapeamento simplificado por preferência de mercado
            def map_contract_preference(contract):
                for key, value in contract_types.items():
                    if key in str(contract):
                        return value
                return 0
                
            self.df['ranking_contratacao'] = self.df['tipo_contratacao'].apply(map_contract_preference)
        
        # Features baseadas em engajamento (identificado na análise como importante)
        if 'engajamento_positivo' in self.df.columns:
            # Taxa de engajamento por recrutador
            recruiter_engagement = self.df.groupby('recrutador')['engajamento_positivo'].mean().to_dict()
            self.df['taxa_engajamento_recrutador'] = self.df['recrutador'].map(recruiter_engagement)
            
        # Features baseadas nos comentários
        if 'comentario' in self.df.columns:
            # Quantidade de palavras no comentário
            self.df['qtd_palavras_comentario'] = self.df['comentario'].apply(
                lambda x: len(str(x).split()) if not pd.isna(x) else 0
            )
            
            # Presença de palavras-chave identificadas na análise
            keywords = {
                'inicio_projeto': ['inicio', 'projeto', 'começo', 'começar', 'início'],
                'cliente': ['cliente', 'contratante'],
                'valor': ['valor', 'salário', 'remuneração', 'proposta', 'financeiro'],
                'outra_proposta': ['outra proposta', 'outra oferta', 'outro processo'],
                'aceitar': ['aceita', 'aceitou', 'aceitar', 'topou', 'aprovou']
            }
            
            for key, words in keywords.items():
                self.df[f'menciona_{key}'] = self.df['comentario'].apply(
                    lambda x: any(word in str(x).lower() for word in words) if not pd.isna(x) else False
                )
                
        # Combinar features para detectar perfis mais propensos à desistência
        if all(col in self.df.columns for col in ['menciona_outra_proposta', 'menciona_valores', 'engajamento_positivo']):
            self.df['risco_desistencia'] = (
                (self.df['menciona_outra_proposta']) | 
                (self.df['menciona_valores'] & ~self.df['engajamento_positivo'])
            ).astype(int)
            
        print("✅ Features agregadas criadas com base nos insights da análise")
        
    def prepare_features_for_ml(self):
        """Prepara features finais para modelos ML com base nos insights da análise"""
        # Verificar quais features estão disponíveis
        available_columns = set(self.df.columns)
        
        # Selecionar features numéricas (baseado nos insights da análise)
        numeric_features_base = [
            'dias_no_pipeline', 'mes_candidatura', 'ano_candidatura', 
            'dia_semana_candidatura', 'trimestre_candidatura', 'fim_do_mes', 'inicio_do_mes',
            'sentimento_comentario', 'pretensao_salarial_extraida', 'disponibilidade_dias',
            'tamanho_comentario', 'qtd_palavras_comentario',
            'taxa_sucesso_recrutador', 'total_candidatos_recrutador',
            'popularidade_area', 'tempo_medio_pipeline_area', 'taxa_sucesso_area',
            'taxa_sucesso_contratacao', 'ranking_contratacao', 'taxa_engajamento_recrutador'
        ]
        
        # Filtrar apenas as colunas que realmente existem
        numeric_features = [col for col in numeric_features_base if col in available_columns]
        
        # Selecionar features categóricas encoded
        categorical_features_base = [
            'recrutador_encoded', 'area_vaga_encoded', 
            'senioridade_vaga_encoded', 'modalidade_encoded', 'tipo_contratacao_encoded'
        ]
        categorical_features = [col for col in categorical_features_base if col in available_columns]
        
        # Selecionar features one-hot (se existirem)
        onehot_features = [col for col in available_columns if 
                          col.startswith('area_vaga_') or 
                          col.startswith('senioridade_vaga_') or 
                          col.startswith('tipo_contratacao_')]
        
        # Selecionar features booleanas importantes identificadas na análise
        boolean_features_base = [
            'menciona_ingles', 'menciona_cliente', 'menciona_projeto',
            'menciona_contratacao', 'menciona_valores', 'menciona_desistencia',
            'menciona_outra_proposta', 'engajamento_positivo', 'fit_cultural',
            'risco_desistencia', 'recrutador_alto_desempenho', 'tempo_acima_media',
            'menciona_inicio_projeto', 'menciona_cliente', 'menciona_valor',
            'menciona_outra_proposta', 'menciona_aceitar'
        ]
        # Adicionar features para skills específicas
        boolean_features_base.extend([f'menciona_{skill}' for skill in ['java', 'python', 'sap', 'cloud', 'agile']])
        
        boolean_features = [col for col in boolean_features_base if col in available_columns]
        
        # Combinar todas as features
        all_features = numeric_features + categorical_features + onehot_features + boolean_features
        
        print(f"✅ Selecionadas {len(all_features)} features para o modelo")
        
        # Criar dataset final
        features_df = self.df[all_features].copy()
        
        # Tratar valores faltantes
        for col in numeric_features:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(features_df[col].median())
                
        for col in boolean_features:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(False).astype(int)  # Converter para 0/1
        
        # Verificar colunas com variância zero e remover
        zero_var_cols = []
        for col in features_df.columns:
            try:
                unique_values = features_df[col].nunique()
                if unique_values <= 1:
                    zero_var_cols.append(col)
            except Exception as e:
                print(f"⚠️ Erro ao verificar variância da coluna {col}: {e}")
        
        if zero_var_cols:
            print(f"⚠️ Removidas {len(zero_var_cols)} colunas com variância zero")
            features_df = features_df.drop(columns=zero_var_cols)
        
        # Normalizar features numéricas (importante para algoritmos como SVM e redes neurais)
        numeric_features_existing = [col for col in numeric_features if col in features_df.columns]
        if numeric_features_existing:
            features_df[numeric_features_existing] = self.scaler.fit_transform(features_df[numeric_features_existing])
            
            # Salvar o scaler para uso futuro
            if not os.path.exists('models'):
                os.makedirs('models')
            with open('models/feature_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # Verificar target variable
        if 'target_sucesso' in self.df.columns:
            target = self.df['target_sucesso']
            print(f"✅ Target variable: {target.value_counts().to_dict()}")
        else:
            print("⚠️ Target variable 'target_sucesso' não encontrada")
            target = None
        
        return features_df, target
        
    def save_processed_data(self):
        """Salva dados processados"""
        X, y = self.prepare_features_for_ml()
        
        # Criar diretórios se não existirem
        if not os.path.exists('data/processed'):
            os.makedirs('data/processed')
            
        # Salvar features e target
        X.to_csv('data/processed/features_engineered.csv', index=False)
        if y is not None:
            y.to_csv('data/processed/target_variable.csv', index=False)
        
        # Salvar dados completos processados
        self.df.to_csv('data/processed/complete_processed_data.csv', index=False)
        
        # Salvar os encoders para uso futuro
        if not os.path.exists('models'):
            os.makedirs('models')
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        print("✅ Dados processados salvos")
        return X, y

    def run_pipeline(self):
        """Executa todo o pipeline de feature engineering"""
        print("\n" + "="*50)
        print("🚀 INICIANDO PIPELINE DE FEATURE ENGINEERING")
        print("="*50)
        
        # Etapa 1: Features temporais
        self.extract_temporal_features()
        
        # Etapa 2: Features de texto dos comentários
        self.extract_text_features()
        
        # Etapa 3: Features categóricas
        self.extract_categorical_features()
        
        # Etapa 4: Features agregadas
        self.create_aggregated_features()
        
        # Etapa 5: Preparação final e salvamento
        X, y = self.save_processed_data()
        
        print("\n" + "="*50)
        print("✅ PIPELINE DE FEATURE ENGINEERING CONCLUÍDO")
        print("="*50)
        
        # Estatísticas finais
        print(f"\nFeatures finais: {X.shape[1]} features para {X.shape[0]} amostras")
        if y is not None:
            print(f"Distribuição do target: {y.value_counts().to_dict()}")
        else:
            print("⚠️ Não foi possível criar a variável target")
            
        return X, y

# Executar feature engineering
if __name__ == "__main__":
    feature_engineer = DecisionFeatureEngineer()
    X, y = feature_engineer.run_pipeline()
    
    # Mostrar algumas das features mais importantes
    print("\nPrincipais features criadas:")
    important_features = [col for col in X.columns if any(
        term in col for term in ['engajamento', 'sucesso', 'risco', 'recrutador_alto', 'menciona_'])]
    for feature in important_features[:10]:
        print(f"  - {feature}")

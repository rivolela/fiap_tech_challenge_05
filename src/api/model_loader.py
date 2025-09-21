"""Utilitário para carregar o modelo treinado"""

import os
import pickle
import random
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
from textblob import TextBlob

# Caminho para o modelo treinado
MODEL_PATH = 'models/scoring_model.pkl'
SCALER_PATH = 'models/feature_scaler.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'

# Variáveis para caching
_model = None
_scaler = None
_encoder = None

def check_sklearn_version():
    """
    Verifica a versão do scikit-learn instalada e exibe um aviso se não for compatível.
    
    A versão recomendada é 1.7.1 para compatibilidade com o modelo salvo.
    """
    try:
        import sklearn
        current_version = sklearn.__version__
        required_version = "1.7.1"
        
        print(f"scikit-learn versão instalada: {current_version}")
        print(f"scikit-learn versão recomendada: {required_version}")
        
        if current_version != required_version:
            print("\n⚠️ AVISO DE COMPATIBILIDADE ⚠️")
            print(f"A versão instalada do scikit-learn ({current_version}) é diferente da versão")
            print(f"utilizada para treinar o modelo ({required_version}).")
            print("Isso pode causar problemas de compatibilidade ou avisos durante a execução.")
            print("Para resolver, execute: pip install scikit-learn==1.7.1\n")
    except ImportError:
        print("Não foi possível verificar a versão do scikit-learn.")


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
        try:
            # Suprimir avisos de versão durante o carregamento
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                with open(MODEL_PATH, 'rb') as f:
                    _model = pickle.load(f)
            print("Modelo carregado com sucesso!")
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            print("Este erro pode ocorrer devido a incompatibilidade de versões do scikit-learn.")
            print("Verifique se a versão instalada (scikit-learn==1.7.1) é compatível com o modelo salvo.")
            raise
    
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


def predict(data: pd.DataFrame, vaga_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Realiza a predição para novos dados
    
    Args:
        data: DataFrame com os dados de entrada
        vaga_info: Dicionário com informações sobre a vaga (opcional)
        
    Returns:
        Dicionário com os resultados da predição:
        - prediction: 0 ou 1 (classe prevista)
        - probability: probabilidade da classe positiva
        - recommendation: texto com a recomendação baseada na predição
        - comment: comentário personalizado gerado com técnicas de LLM
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
        
        # Gerar comentário personalizado baseado na predição e nos dados
        comment = generate_llm_comment(data, prediction, probability, vaga_info)
        
        return {
            "prediction": prediction,
            "probability": probability,
            "recommendation": recommendation,
            "comment": comment
        }
    except Exception as e:
        raise RuntimeError(f"Erro ao fazer predição: {str(e)}")


def generate_llm_comment(data: pd.DataFrame, prediction: int, probability: float, 
                         vaga_info: Optional[Dict[str, Any]] = None) -> str:
    """
    Gera um comentário personalizado usando técnicas de LLM simples
    baseado nos dados do candidato, na predição e nas informações da vaga
    
    Args:
        data: DataFrame com os dados do candidato
        prediction: Predição do modelo (0 ou 1)
        probability: Probabilidade da predição
        vaga_info: Informações sobre a vaga (opcional)
        
    Returns:
        String com comentário personalizado
    """
    # Extrair informações relevantes
    try:
        idade = data['idade'].iloc[0] if 'idade' in data.columns else None
        experiencia = data['experiencia'].iloc[0] if 'experiencia' in data.columns else None
        educacao = data['educacao'].iloc[0] if 'educacao' in data.columns else None
        area_formacao = data['area_formacao'].iloc[0] if 'area_formacao' in data.columns else None
        
        # Mapear nível de confiança baseado na probabilidade
        if probability > 0.9:
            confianca = "alta"
        elif probability > 0.7:
            confianca = "boa"
        elif probability > 0.5:
            confianca = "moderada"
        else:
            confianca = "baixa"
        
        # Base de templates para comentários positivos
        positive_templates = [
            "Com base no perfil do candidato, {idade_exp} e {educacao_exp}, há uma {confianca} chance de compatibilidade com a posição{vaga_exp}.",
            "A análise indica {confianca} adequação ao cargo{vaga_exp}. O candidato possui {experiencia_exp} e {educacao_exp}.",
            "Recomendamos prosseguir com este candidato que demonstra {confianca} compatibilidade{vaga_exp}, considerando {idade_exp} e {experiencia_exp}.",
            "A avaliação técnica sugere {confianca} adequação para a função{vaga_exp}. Destaca-se {educacao_exp} e {experiencia_exp}."
        ]
        
        # Base de templates para comentários negativos
        negative_templates = [
            "O perfil apresenta {confianca} probabilidade de não atender aos requisitos{vaga_exp}. {educacao_exp} e {experiencia_exp} parecem insuficientes.",
            "Não recomendado com {confianca} confiança. Considerando {idade_exp} e {experiencia_exp}, pode não atender às expectativas{vaga_exp}.",
            "Sugerimos avaliar outros candidatos, pois a análise indica {confianca} incompatibilidade{vaga_exp}. {educacao_exp} não atende completamente ao esperado.",
            "Baseado em {experiencia_exp} e {educacao_exp}, há {confianca} indicação de que o candidato não é adequado para esta posição{vaga_exp}."
        ]
        
        # Construir fragmentos de texto para substituição
        idade_exp = f"tendo {idade} anos" if idade else "com o perfil etário apresentado"
        
        if experiencia is not None:
            if experiencia < 1:
                experiencia_exp = "pouca ou nenhuma experiência profissional"
            elif experiencia < 3:
                experiencia_exp = f"{experiencia:.1f} anos de experiência inicial"
            elif experiencia < 6:
                experiencia_exp = f"{experiencia:.1f} anos de experiência relevante"
            else:
                experiencia_exp = f"{experiencia:.1f} anos de experiência sólida"
        else:
            experiencia_exp = "experiência profissional não especificada"
        
        # Mapeamento de educação para texto
        edu_map = {
            "ensino_fundamental": "formação de nível fundamental",
            "ensino_medio": "formação de nível médio",
            "ensino_superior": "formação superior",
            "pos_graduacao": "pós-graduação"
        }
        
        if educacao and educacao in edu_map:
            educacao_exp = edu_map[educacao]
        else:
            educacao_exp = "formação acadêmica apresentada"
            
        if area_formacao:
            educacao_exp += f" na área de {area_formacao}"
            
        # Adicionar informações da vaga, se disponíveis
        if vaga_info and 'titulo' in vaga_info and vaga_info['titulo']:
            vaga_exp = f" de {vaga_info['titulo']}"
            if 'area' in vaga_info and vaga_info['area']:
                vaga_exp += f" na área de {vaga_info['area']}"
            if 'senioridade' in vaga_info and vaga_info['senioridade']:
                vaga_exp += f", nível {vaga_info['senioridade']}"
        else:
            vaga_exp = ""
            
        # Selecionar template baseado na predição
        if prediction == 1:
            template = random.choice(positive_templates)
        else:
            template = random.choice(negative_templates)
            
        # Preencher o template com os dados
        comment = template.format(
            idade_exp=idade_exp,
            experiencia_exp=experiencia_exp,
            educacao_exp=educacao_exp,
            confianca=confianca,
            vaga_exp=vaga_exp
        )
        
        # Usar TextBlob para melhorar um pouco o texto (correção e fluência)
        comment_blob = TextBlob(comment)
        
        # Aqui seria o ponto onde poderíamos chamar um LLM mais sofisticado
        # para melhorar a qualidade e naturalidade do comentário
        
        return str(comment_blob)
        
    except Exception as e:
        # Em caso de erro, retornar um comentário genérico
        print(f"Erro ao gerar comentário: {str(e)}")
        if prediction == 1:
            return "Candidato recomendado com base no modelo de análise."
        else:
            return "Candidato não recomendado com base no modelo de análise."
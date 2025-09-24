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
    Gera um comentário personalizado usando técnicas de LLM avançado
    baseado nos dados do candidato, na predição e nas informações da vaga.
    Inclui análise detalhada de compatibilidade, pontos fortes e áreas de atenção.
    
    Args:
        data: DataFrame com os dados do candidato
        prediction: Predição do modelo (0 ou 1)
        probability: Probabilidade da predição
        vaga_info: Informações sobre a vaga (opcional)
        
    Returns:
        String com comentário personalizado e análise detalhada
    """
    # Log para diagnóstico
    print("DEBUG - generate_llm_comment chamado")
    print(f"DEBUG - data.columns: {data.columns.tolist()}")
    if 'area_formacao' in data.columns:
        print(f"DEBUG - area_formacao: '{data['area_formacao'].iloc[0]}', tipo: {type(data['area_formacao'].iloc[0])}")
    if vaga_info:
        print(f"DEBUG - vaga_info: {vaga_info}")
    
    # Extrair informações relevantes
    try:
        idade = data['idade'].iloc[0] if 'idade' in data.columns else None
        experiencia = data['experiencia'].iloc[0] if 'experiencia' in data.columns else None
        educacao = data['educacao'].iloc[0] if 'educacao' in data.columns else None
        area_formacao = data['area_formacao'].iloc[0] if 'area_formacao' in data.columns else None
        tempo_desempregado = data['tempo_desempregado'].iloc[0] if 'tempo_desempregado' in data.columns else None
        habilidades = data['habilidades'].iloc[0] if 'habilidades' in data.columns else []
        cargo_anterior = data['cargo_anterior'].iloc[0] if 'cargo_anterior' in data.columns else None
        anos_estudo = data['anos_estudo'].iloc[0] if 'anos_estudo' in data.columns else None
        
        # Extrair informações da vaga
        vaga_titulo = vaga_info.get('titulo', '') if vaga_info else ''
        vaga_area = vaga_info.get('area', '') if vaga_info else ''
        vaga_senioridade = vaga_info.get('senioridade', '') if vaga_info else ''
        
        # Mais logs
        print(f"DEBUG - area_formacao extraído: '{area_formacao}'")
        print(f"DEBUG - vaga_area extraído: '{vaga_area}'")
        
        # Mapear nível de confiança baseado na probabilidade
        if probability > 0.9:
            confianca = "alta"
        elif probability > 0.7:
            confianca = "boa"
        elif probability > 0.5:
            confianca = "moderada"
        else:
            confianca = "baixa"
            
        # Mapeamento de educação para texto
        edu_map = {
            "ensino_fundamental": "formação de nível fundamental",
            "ensino_medio": "formação de nível médio",
            "ensino_superior": "formação superior",
            "pos_graduacao": "pós-graduação"
        }
        
        # Construir fragmentos de informação
        educacao_texto = edu_map.get(educacao, "formação acadêmica") if educacao else "formação não especificada"
        experiencia_texto = f"{experiencia:.1f} anos" if experiencia is not None else "não especificada"
        
        # Análise de compatibilidade de área
        match_area = False
        print(f"DEBUG - Verificando match de área. area_formacao: '{area_formacao}', vaga_area: '{vaga_area}'")
        
        # Primeiro, garantir que as strings são strings válidas e não "string" ou "None"
        if area_formacao is not None and isinstance(area_formacao, str):
            area_formacao = area_formacao.strip()
            if area_formacao.lower() in ["string", "none", ""]:
                area_formacao = None
                
        if vaga_area is not None and isinstance(vaga_area, str):
            vaga_area = vaga_area.strip()
            if vaga_area.lower() in ["string", "none", ""]:
                vaga_area = None
                
        print(f"DEBUG - Após sanitização: area_formacao: '{area_formacao}', vaga_area: '{vaga_area}'")
        
        if area_formacao is not None and vaga_area is not None:
            match_area = (area_formacao.lower() == vaga_area.lower() or 
                         area_formacao.lower() in vaga_area.lower() or 
                         vaga_area.lower() in area_formacao.lower())
            print(f"DEBUG - Resultado do match: {match_area}")
            
        # Análise de senioridade
        senioridade_adequada = True
        razao_senioridade = ""
        
        if vaga_senioridade and experiencia is not None:
            if vaga_senioridade.lower() in ['senior', 'sênior'] and experiencia < 6:
                senioridade_adequada = False
                razao_senioridade = f"A vaga requer senioridade senior, mas o candidato possui {experiencia_texto} de experiência (geralmente esperado 6+ anos)."
            elif vaga_senioridade.lower() in ['pleno', 'pl'] and experiencia < 3:
                senioridade_adequada = False
                razao_senioridade = f"A vaga requer senioridade pleno, mas o candidato possui {experiencia_texto} de experiência (geralmente esperado 3+ anos)."
            elif vaga_senioridade.lower() in ['junior', 'jr'] and experiencia > 5:
                senioridade_adequada = True
                razao_senioridade = f"O candidato possui {experiencia_texto} de experiência, superior ao geralmente esperado para uma vaga junior."
        
        # Gerar análise detalhada
        if prediction == 1:
            # Comentário positivo com análise detalhada
            pontos_positivos = []
            if match_area:
                if area_formacao is not None and vaga_area is not None:
                    pontos_positivos.append(f"Formação em {area_formacao} compatível com a área da vaga ({vaga_area})")
                else:
                    pontos_positivos.append("Perfil profissional alinhado com os requisitos da vaga")
            if experiencia is not None and experiencia > 3:
                pontos_positivos.append(f"Experiência sólida de {experiencia_texto}")
            if tempo_desempregado is not None and tempo_desempregado < 0.5:
                pontos_positivos.append("Curto período sem emprego, indicando atualização profissional")
            if len(habilidades) > 0:
                habs_texto = ", ".join(habilidades[:3])
                pontos_positivos.append(f"Competências relevantes: {habs_texto}")
            
            pontos_atencao = []
            if not match_area:
                pontos_atencao.append(f"Formação em {area_formacao} diferente da área da vaga ({vaga_area})")
            if not senioridade_adequada:
                pontos_atencao.append(razao_senioridade)
                
            # Construir o texto da análise
            analise = f"Análise indica {confianca} compatibilidade com a vaga"
            if vaga_titulo:
                analise += f" de {vaga_titulo}"
            if vaga_area:
                analise += f" na área de {vaga_area}"
            if vaga_senioridade:
                analise += f", nível {vaga_senioridade}"
            analise += ".\n\n"
            
            if pontos_positivos:
                analise += "Pontos fortes:\n"
                for ponto in pontos_positivos:
                    analise += f"• {ponto}\n"
                analise += "\n"
                
            if pontos_atencao:
                analise += "Áreas de atenção:\n"
                for ponto in pontos_atencao:
                    analise += f"• {ponto}\n"
                    
            return analise.strip()
            
        else:
            # Comentário negativo com análise detalhada
            razoes_negativas = []
            
            if not match_area:
                # Verificar se ambos estão definidos antes de adicionar esta razão
                if area_formacao is not None and vaga_area is not None:
                    area_form_texto = area_formacao
                    area_vaga_texto = vaga_area
                    razoes_negativas.append(f"Incompatibilidade entre a formação do candidato ({area_form_texto}) e a área da vaga ({area_vaga_texto})")
                elif area_formacao is None and vaga_area is not None:
                    razoes_negativas.append(f"Candidato não especificou área de formação para vaga na área de {vaga_area}")
                elif area_formacao is not None and vaga_area is None:
                    razoes_negativas.append(f"Candidato com formação em {area_formacao}, mas área da vaga não especificada")
            
            if not senioridade_adequada:
                razoes_negativas.append(razao_senioridade)
                
            if experiencia is not None and experiencia < 2:
                razoes_negativas.append(f"Experiência limitada de apenas {experiencia_texto}")
                
            if tempo_desempregado is not None and tempo_desempregado > 1:
                razoes_negativas.append(f"Período de {tempo_desempregado:.1f} anos sem emprego formal na área")
            
            # Construir o texto da análise
            analise = f"O perfil apresenta {confianca} probabilidade de não atender completamente aos requisitos"
            if vaga_titulo:
                analise += f" da posição de {vaga_titulo}"
            if vaga_area:
                analise += f" na área de {vaga_area}"
            if vaga_senioridade:
                analise += f", nível {vaga_senioridade}"
            analise += ".\n\n"
            
            if razoes_negativas:
                analise += "Principais considerações:\n"
                for razao in razoes_negativas:
                    analise += f"• {razao}\n"
                    
            if prediction == 0 and probability < 0.3:
                analise += "\nSugerimos avaliar outros candidatos mais alinhados com os requisitos da posição."
            elif prediction == 0 and probability >= 0.3:
                analise += "\nApesar da recomendação negativa, o candidato possui alguns pontos que podem ser considerados em uma segunda análise, caso não haja outros candidatos adequados."
                
            return analise.strip()
        
    except Exception as e:
        # Em caso de erro, retornar um comentário genérico
        print(f"Erro ao gerar comentário: {str(e)}")
        if prediction == 1:
            return "Candidato recomendado com base no modelo de análise."
        else:
            return "Candidato não recomendado com base no modelo de análise."
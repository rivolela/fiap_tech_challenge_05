"""Utilitários para pré-processamento de dados na API"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union

def preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Pré-processa os dados de entrada para predição
    
    Args:
        data: Dicionário com os dados do candidato
        
    Returns:
        DataFrame pronto para ser usado pelo modelo
    """
    # Converter para DataFrame
    df = pd.DataFrame([data])
    
    # Processar tipos de dados e formatos
    _process_numeric_fields(df)
    _process_categorical_fields(df)
    _process_list_fields(df)
    
    # Processar campos relacionados à vaga
    _process_job_fields(df)
    
    # Adicionar features calculadas, se necessário
    _add_calculated_features(df)
    
    # Verificar se há colunas faltantes que o modelo espera
    from src.api.model_loader import get_feature_list
    required_features = get_feature_list()
    
    # Adicionar colunas faltantes com valores padrão
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Garantir a mesma ordem de colunas que o modelo espera
    df = df[required_features]
    
    return df


def preprocess_batch(batch_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Pré-processa um lote de dados para predição em batch
    
    Args:
        batch_data: Lista de dicionários com os dados dos candidatos
        
    Returns:
        DataFrame pronto para ser usado pelo modelo
    """
    # Processar cada item individualmente e concatenar os resultados
    processed_dfs = []
    
    for item in batch_data:
        processed_df = preprocess_input(item)
        processed_dfs.append(processed_df)
    
    return pd.concat(processed_dfs, ignore_index=True)


def _process_numeric_fields(df: pd.DataFrame) -> None:
    """
    Processa campos numéricos no DataFrame
    
    Args:
        df: DataFrame a ser processado
    """
    numeric_fields = ['idade', 'experiencia', 'tempo_desempregado', 
                     'anos_estudo', 'salario_anterior', 'anos_ultimo_emprego']
    
    # Converter campos para numérico, se existirem
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce')
            
    # Preencher valores nulos com a média ou um valor padrão
    for field in numeric_fields:
        if field in df.columns and df[field].isnull().any():
            # Valores padrão para cada tipo de campo
            default_values = {
                'idade': 30,
                'experiencia': 5,
                'tempo_desempregado': 0,
                'anos_estudo': 12,
                'salario_anterior': 0,
                'anos_ultimo_emprego': 0
            }
            df[field].fillna(default_values.get(field, 0), inplace=True)


def _process_categorical_fields(df: pd.DataFrame) -> None:
    """
    Processa campos categóricos no DataFrame
    
    Args:
        df: DataFrame a ser processado
    """
    # Mapeamento para padronização de alguns valores categóricos
    education_mapping = {
        'fundamental': 'ensino_fundamental',
        'medio': 'ensino_medio',
        'superior': 'ensino_superior',
        'graduacao': 'ensino_superior',
        'pos': 'pos_graduacao',
        'mestrado': 'pos_graduacao',
        'doutorado': 'pos_graduacao'
    }
    
    # Aplicar mapeamentos para padronização
    if 'educacao' in df.columns:
        df['educacao'] = df['educacao'].str.lower()
        df['educacao'] = df['educacao'].map(lambda x: education_mapping.get(x, x))


def _process_list_fields(df: pd.DataFrame) -> None:
    """
    Processa campos de lista (como habilidades) no DataFrame
    
    Args:
        df: DataFrame a ser processado
    """
    # Processar listas de habilidades
    if 'habilidades' in df.columns:
        # Se o campo já for uma lista, não faz nada
        if isinstance(df['habilidades'].iloc[0], list):
            pass
        # Se for string, tenta converter para lista
        elif isinstance(df['habilidades'].iloc[0], str):
            df['habilidades'] = df['habilidades'].apply(
                lambda x: [item.strip() for item in x.split(',')]
            )
        else:
            # Se não for lista nem string, converte para lista vazia
            df['habilidades'] = [[]]
        
        # Criar colunas one-hot para habilidades comuns
        common_skills = ['python', 'java', 'javascript', 'html', 'css', 'sql', 
                        'r', 'c++', 'c#', 'php', 'ruby', 'swift', 'kotlin',
                        'analise_dados', 'machine_learning', 'ia', 'estatistica']
        
        for skill in common_skills:
            col_name = f'skill_{skill}'
            df[col_name] = df['habilidades'].apply(
                lambda skills: 1 if skill in [s.lower() for s in skills] else 0
            )


def _add_calculated_features(df: pd.DataFrame) -> None:
    """
    Adiciona features calculadas a partir dos dados existentes
    
    Args:
        df: DataFrame a ser processado
    """
    # Exemplo: calcular razão experiência/idade (indicador de carreira)
    if 'idade' in df.columns and 'experiencia' in df.columns:
        # Evitar divisão por zero
        df['idade'] = df['idade'].apply(lambda x: max(1, x))
        df['experiencia_por_idade'] = df['experiencia'] / df['idade']
        
    # Calcular tempo de experiência não considerando desemprego
    if 'experiencia' in df.columns and 'tempo_desempregado' in df.columns:
        df['experiencia_ativa'] = df['experiencia'] - df['tempo_desempregado'].fillna(0)
        

def _process_job_fields(df: pd.DataFrame) -> None:
    """
    Processa campos relacionados à vaga no DataFrame
    
    Args:
        df: DataFrame a ser processado
    """
    # Processar o ID da vaga
    if 'vaga_id' in df.columns:
        df['vaga_id'] = df['vaga_id'].fillna('desconhecido')
    
    # Processar título da vaga
    if 'vaga_titulo' in df.columns:
        df['vaga_titulo'] = df['vaga_titulo'].fillna('').str.lower()
    
    # Processar área da vaga
    if 'vaga_area' in df.columns:
        df['vaga_area'] = df['vaga_area'].fillna('').str.lower()
        
        # Mapear áreas para categorias padrão
        area_mapping = {
            'ti': 'tecnologia',
            'tecnologia': 'tecnologia',
            'tech': 'tecnologia',
            'vendas': 'comercial',
            'comercial': 'comercial',
            'marketing': 'marketing',
            'recursos humanos': 'rh',
            'rh': 'rh',
            'financeiro': 'financeiro',
            'finanças': 'financeiro',
            'contabilidade': 'financeiro',
            'administrativo': 'administrativo',
            'adm': 'administrativo',
            'engenharia': 'engenharia'
        }
        
        df['vaga_area'] = df['vaga_area'].apply(lambda x: area_mapping.get(str(x).lower(), x) if pd.notna(x) else x)
    
    # Processar senioridade da vaga
    if 'vaga_senioridade' in df.columns:
        df['vaga_senioridade'] = df['vaga_senioridade'].fillna('').str.lower()
        
        # Mapear senioridade para valores padrão
        senioridade_mapping = {
            'junior': 'junior',
            'júnior': 'junior',
            'jr': 'junior',
            'pleno': 'pleno',
            'pl': 'pleno',
            'senior': 'senior',
            'sênior': 'senior',
            'sr': 'senior',
            'especialista': 'senior',
            'trainee': 'junior',
            'estagiário': 'estagio',
            'estagiario': 'estagio',
            'estágio': 'estagio'
        }
        
        df['vaga_senioridade'] = df['vaga_senioridade'].apply(lambda x: senioridade_mapping.get(str(x).lower(), x) if pd.notna(x) else x)
    
    # Calcular match entre área de formação e área da vaga
    if 'area_formacao' in df.columns and 'vaga_area' in df.columns:
        df['match_area'] = df.apply(
            lambda row: 1 if pd.notna(row['area_formacao']) and pd.notna(row['vaga_area']) and 
                             str(row['area_formacao']).lower() == str(row['vaga_area']).lower() 
                        else 0, 
            axis=1
        )
    
    # Verificar se existe a coluna experiencia_ativa antes de aplicar a função
    if 'experiencia_ativa' in df.columns:
        df['experiencia_ativa'] = df['experiencia_ativa'].apply(lambda x: max(0, x))
    
    # Outras features calculadas podem ser adicionadas conforme necessário
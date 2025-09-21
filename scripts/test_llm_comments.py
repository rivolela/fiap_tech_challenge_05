"""
Script para testar a geração de comentários pelo LLM simples.

Este script importa a função de geração de comentários e a executa
com diferentes perfis de candidatos e vagas para demonstrar a
funcionalidade.
"""

import pandas as pd
from src.api.model_loader import generate_llm_comment
import json

def main():
    """Executa os testes de geração de comentários"""
    print("Testando geração de comentários com LLM simples...\n")
    
    # Caso 1: Candidato recomendado para vaga de tecnologia
    print("=== Caso 1: Candidato recomendado para desenvolvedor Python ===")
    data_case1 = pd.DataFrame({
        'idade': [28],
        'experiencia': [5],
        'educacao': ['ensino_superior'],
        'area_formacao': ['tecnologia'],
        'tempo_desempregado': [0.5]
    })
    
    vaga_case1 = {
        'id': 'vaga-123',
        'titulo': 'Desenvolvedor Python',
        'area': 'tecnologia',
        'senioridade': 'pleno'
    }
    
    comment1 = generate_llm_comment(data_case1, 1, 0.85, vaga_case1)
    print(f"Comentário: {comment1}")
    print("\n")
    
    # Caso 2: Candidato não recomendado para vaga de marketing
    print("=== Caso 2: Candidato não recomendado para marketing ===")
    data_case2 = pd.DataFrame({
        'idade': [22],
        'experiencia': [1.2],
        'educacao': ['ensino_medio'],
        'area_formacao': ['administracao'],
        'tempo_desempregado': [2.0]
    })
    
    vaga_case2 = {
        'id': 'vaga-456',
        'titulo': 'Analista de Marketing',
        'area': 'marketing',
        'senioridade': 'junior'
    }
    
    comment2 = generate_llm_comment(data_case2, 0, 0.35, vaga_case2)
    print(f"Comentário: {comment2}")
    print("\n")
    
    # Caso 3: Candidato experiente para vaga sênior
    print("=== Caso 3: Candidato experiente para vaga sênior ===")
    data_case3 = pd.DataFrame({
        'idade': [42],
        'experiencia': [15],
        'educacao': ['pos_graduacao'],
        'area_formacao': ['engenharia'],
        'tempo_desempregado': [0.0]
    })
    
    vaga_case3 = {
        'id': 'vaga-789',
        'titulo': 'Gerente de Projetos',
        'area': 'engenharia',
        'senioridade': 'senior'
    }
    
    comment3 = generate_llm_comment(data_case3, 1, 0.93, vaga_case3)
    print(f"Comentário: {comment3}")
    print("\n")
    
    # Caso 4: Candidato com dados incompletos
    print("=== Caso 4: Candidato com dados incompletos ===")
    data_case4 = pd.DataFrame({
        'idade': [35],
        'experiencia': [3]
    })
    
    vaga_case4 = {
        'id': 'vaga-101',
        'titulo': 'Analista de Dados',
    }
    
    comment4 = generate_llm_comment(data_case4, 0, 0.48, vaga_case4)
    print(f"Comentário: {comment4}")
    print("\n")
    
    print("Testes concluídos!")

if __name__ == "__main__":
    main()
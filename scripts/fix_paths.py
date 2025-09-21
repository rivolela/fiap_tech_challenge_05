#!/usr/bin/env python
# fix_paths.py - Script para corrigir caminhos no projeto

import os
import sys

def fix_model_loader_paths():
    """Modifica os caminhos no arquivo model_loader.py para usar caminhos absolutos"""
    
    model_loader_path = os.path.join('src', 'api', 'model_loader.py')
    
    if not os.path.exists(model_loader_path):
        print(f"ERRO: Arquivo {model_loader_path} não encontrado!")
        return False
    
    # Ler o conteúdo do arquivo
    with open(model_loader_path, 'r') as file:
        content = file.read()
    
    # Verificar se já foi corrigido
    if 'os.path.join(os.path.dirname(__file__)' in content:
        print("Arquivo já parece estar corrigido com caminhos relativos.")
        return True
    
    # Substituições a serem feitas
    replacements = [
        ("MODEL_PATH = 'models/scoring_model.pkl'",
         "MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'scoring_model.pkl')"),
        ("SCALER_PATH = 'models/feature_scaler.pkl'",
         "SCALER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'feature_scaler.pkl')"),
        ("ENCODER_PATH = 'models/label_encoder.pkl'",
         "ENCODER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'label_encoder.pkl')")
    ]
    
    # Aplicar as substituições
    for old, new in replacements:
        content = content.replace(old, new)
    
    # Salvar o arquivo atualizado
    with open(model_loader_path, 'w') as file:
        file.write(content)
    
    print(f"Arquivo {model_loader_path} atualizado com caminhos absolutos.")
    return True


def fix_get_feature_list():
    """Corrige a função get_feature_list para usar caminhos absolutos"""
    
    model_loader_path = os.path.join('src', 'api', 'model_loader.py')
    
    if not os.path.exists(model_loader_path):
        print(f"ERRO: Arquivo {model_loader_path} não encontrado!")
        return False
    
    # Ler o conteúdo do arquivo
    with open(model_loader_path, 'r') as file:
        content = file.readlines()
    
    in_function = False
    modified = False
    
    for i, line in enumerate(content):
        # Identifica o início da função get_feature_list
        if 'def get_feature_list()' in line:
            in_function = True
        
        # Dentro da função, procura a linha com o caminho do CSV
        if in_function and "df = pd.read_csv('data/processed/complete_processed_data.csv" in line:
            content[i] = line.replace(
                "df = pd.read_csv('data/processed/complete_processed_data.csv", 
                "df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed', 'complete_processed_data.csv'"
            )
            modified = True
        
        # Identifica o final da função
        if in_function and line.strip() == '':
            in_function = False
    
    # Se houve modificações, salva o arquivo
    if modified:
        with open(model_loader_path, 'w') as file:
            file.writelines(content)
        print(f"Função get_feature_list em {model_loader_path} atualizada com caminhos absolutos.")
    else:
        print("Função get_feature_list já parece estar usando caminhos absolutos ou não foi encontrada.")
    
    return modified


def main():
    """Função principal que executa todas as correções"""
    print("Iniciando correção de caminhos no projeto...\n")
    
    # Adicionar a raiz do projeto ao sys.path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)
    sys.path.insert(0, root_dir)
    
    # Corrigir caminhos
    fix_model_loader_paths()
    fix_get_feature_list()
    
    print("\nCorreções concluídas!")


if __name__ == "__main__":
    main()
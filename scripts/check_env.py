#!/usr/bin/env python3
"""
Script para verificar a configuração do ambiente Python
"""

import sys
import pkg_resources

def check_environment():
    """Verifica se todas as bibliotecas necessárias estão instaladas"""
    
    print("=== VERIFICAÇÃO DO AMBIENTE PYTHON ===")
    print(f"Versão do Python: {sys.version}")
    print(f"Executável: {sys.executable}")
    print()
    
    # Lista de pacotes essenciais
    required_packages = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'plotly',
        'jupyter',
        'ipykernel',
        'scikit-learn',
        'scipy',
        'mlflow'
    ]
    
    print("=== VERIFICAÇÃO DOS PACOTES ===")
    
    all_installed = True
    for package in required_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"✅ {package}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package}: NÃO INSTALADO")
            all_installed = False
    
    print()
    if all_installed:
        print("🎉 Todos os pacotes essenciais estão instalados!")
        print("✅ Ambiente pronto para análise de dados!")
    else:
        print("⚠️  Alguns pacotes estão faltando.")
        print("Execute: pip install -r requirements.txt")
    
    return all_installed

if __name__ == "__main__":
    check_environment()

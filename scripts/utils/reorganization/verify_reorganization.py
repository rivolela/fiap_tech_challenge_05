#!/usr/bin/env python3
"""
Script para verificar se a reorganização do projeto foi bem-sucedida.
Executa uma série de verificações para garantir que todos os arquivos
foram movidos corretamente e que o projeto continua funcionando.
"""

import os
import sys
from pathlib import Path
import importlib
import subprocess
import json

# Configuração
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def check_directory_structure():
    """Verifica se a estrutura de diretórios esperada existe."""
    print("Verificando estrutura de diretórios...")
    
    expected_dirs = [
        '.github/workflows',
        'config/docker/api',
        'config/docker/dashboard',
        'config/nginx',
        'config/render',
        'logs',
        'scripts/deployment',
        'scripts/monitoring',
        'scripts/utils',
        'src',
        'tests',
    ]
    
    all_exist = True
    for dir_path in expected_dirs:
        full_path = PROJECT_ROOT / dir_path
        if not full_path.exists() or not full_path.is_dir():
            print(f"  ERRO: Diretório não encontrado: {dir_path}")
            all_exist = False
        else:
            print(f"  OK: {dir_path}")
    
    return all_exist

def check_moved_files():
    """Verifica se os arquivos foram movidos para os locais corretos."""
    print("Verificando arquivos movidos...")
    
    expected_files = {
        'config/docker/api/config/docker/api/Dockerfile': True,
        'config/docker/dashboard/config/docker/api/Dockerfile': True,
        'config/docker/config/docker/docker-compose.yml': True,
        'config/nginx/nginx.conf': True,
        'config/render/config/render/render.yaml': True,
        'config/render/config/render/Procfile': True,
        'scripts/utils/scripts/utils/debug_api.py': True,
        'scripts/deployment/scripts/deployment/quick_deploy.sh': True,
        'logs/api_logs.log': False,  # Opcional, pode não existir
    }
    
    # Arquivos que não devem mais estar na raiz
    root_files_should_not_exist = [
        'config/docker/api/Dockerfile',
        'config/docker/api/config/docker/dashboard/Dockerfile',
        'config/docker/docker-compose.yml',
        'config/nginx/nginx.conf',
        'config/render/render.yaml',
        'config/render/Procfile',
        'scripts/utils/debug_api.py',
        'scripts/deployment/quick_deploy.sh',
        'test_report.html',
    ]
    
    all_correct = True
    
    # Verifica arquivos movidos
    for file_path, required in expected_files.items():
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            if required:
                print(f"  ERRO: Arquivo esperado não encontrado: {file_path}")
                all_correct = False
            else:
                print(f"  AVISO: Arquivo opcional não encontrado: {file_path}")
        else:
            print(f"  OK: {file_path}")
    
    # Verifica arquivos que não devem estar na raiz
    for file_name in root_files_should_not_exist:
        root_file = PROJECT_ROOT / file_name
        if root_file.exists():
            print(f"  ERRO: Arquivo ainda existe na raiz: {file_name}")
            all_correct = False
    
    return all_correct

def check_imports():
    """Verifica se os imports do projeto ainda funcionam."""
    print("Verificando imports do projeto...")
    
    # Lista de módulos para testar
    modules_to_check = [
        'src.api',
        'src.data',
        'src.features',
        'src.models',
    ]
    
    all_imports_ok = True
    for module_name in modules_to_check:
        try:
            # Adiciona o diretório do projeto ao sys.path se necessário
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
                
            # Tenta importar o módulo
            importlib.import_module(module_name)
            print(f"  OK: {module_name}")
        except ImportError as e:
            print(f"  ERRO: Falha ao importar {module_name}: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def run_tests():
    """Executa os testes do projeto para garantir que tudo ainda funciona."""
    print("Executando testes do projeto...")
    
    test_script = PROJECT_ROOT / 'scripts' / 'run_tests.sh'
    if not test_script.exists():
        print("  AVISO: Script de testes não encontrado")
        return None
    
    try:
        # Executa os testes
        result = subprocess.run(
            ['bash', str(test_script)],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Verifica o resultado
        if result.returncode == 0:
            print("  OK: Todos os testes passaram")
            return True
        else:
            print(f"  ERRO: Alguns testes falharam (código {result.returncode})")
            print(f"  Saída: {result.stdout[:500]}...")  # Mostra parte da saída
            return False
    except Exception as e:
        print(f"  ERRO ao executar testes: {e}")
        return False

def print_summary(results):
    """Imprime um resumo dos resultados da verificação."""
    print("\n=== RESUMO DA VERIFICAÇÃO ===")
    
    all_passed = all(result for result in results.values() if result is not None)
    
    for check, result in results.items():
        status = "OK" if result else "FALHOU" if result is False else "PULADO"
        print(f"  {check}: {status}")
    
    if all_passed:
        print("\nSUCESSO! A reorganização foi concluída com êxito.")
    else:
        print("\nALERTA! A reorganização pode ter problemas que precisam ser corrigidos.")
    
def main():
    results = {}
    
    # Executar verificações
    results['estrutura_diretorios'] = check_directory_structure()
    results['arquivos_movidos'] = check_moved_files()
    results['imports'] = check_imports()
    results['testes'] = run_tests()
    
    # Imprimir resumo
    print_summary(results)
    
    # Retornar código de saída
    return 0 if all(result for result in results.values() if result is not None) else 1

if __name__ == "__main__":
    sys.exit(main())
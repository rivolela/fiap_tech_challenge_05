#!/usr/bin/env python
# debug_paths.py - Ferramenta para depurar problemas de caminhos no Render

import os
import sys
import json
import platform
import importlib.util

def debug_info():
    """Coleta informações de debug sobre o ambiente de execução."""
    info = {
        "environment": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "executable": sys.executable,
            "cwd": os.getcwd(),
            "env_vars": {k: v for k, v in os.environ.items() 
                         if k in ['PATH', 'PYTHONPATH', 'PORT', 'HOME', 'USER']},
        },
        "directories": {},
        "modules": {},
        "files": {}
    }
    
    # Listar diretórios importantes
    important_dirs = ['src', 'models', 'data']
    for d in important_dirs:
        if os.path.exists(d):
            info["directories"][d] = {
                "exists": True,
                "is_dir": os.path.isdir(d),
                "contents": os.listdir(d) if os.path.isdir(d) else None
            }
        else:
            info["directories"][d] = {"exists": False}
    
    # Verificar módulos importantes
    important_modules = ['pandas', 'numpy', 'sklearn', 'fastapi', 'uvicorn']
    for module in important_modules:
        spec = importlib.util.find_spec(module)
        info["modules"][module] = {
            "found": spec is not None,
            "location": spec.origin if spec else None,
            "version": getattr(
                __import__(module) if spec else None,
                "__version__",
                "unknown"
            ) if spec else None
        }
    
    # Verificar arquivos críticos
    critical_files = [
        'models/scoring_model.pkl',
        'models/feature_scaler.pkl',
        'src/api/scoring_api.py'
    ]
    for filepath in critical_files:
        if os.path.exists(filepath):
            info["files"][filepath] = {
                "exists": True,
                "size_bytes": os.path.getsize(filepath),
                "modified": os.path.getmtime(filepath)
            }
        else:
            info["files"][filepath] = {"exists": False}
    
    return info

if __name__ == "__main__":
    # Coletar informações
    info = debug_info()
    
    # Exibir informações formatadas
    print(json.dumps(info, indent=2))
    
    # Salvar em um arquivo
    with open('debug_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\nInformações de debug salvas em debug_info.json")
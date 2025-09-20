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
        "directories": {
            "exists": {
                "/": os.path.exists("/"),
                "/opt": os.path.exists("/opt"),
                "/opt/render": os.path.exists("/opt/render"),
                "/opt/render/project": os.path.exists("/opt/render/project"),
                "/opt/render/project/src": os.path.exists("/opt/render/project/src"),
                "/app": os.path.exists("/app"),
                "./src": os.path.exists("./src"),
                "./models": os.path.exists("./models"),
            }
        },
        "modules": {
            "can_import": {
                "src": importlib.util.find_spec("src") is not None,
                "src.api": importlib.util.find_spec("src.api") is not None,
                "src.api.scoring_api": importlib.util.find_spec("src.api.scoring_api") is not None,
            }
        }
    }
    
    # Listar conteúdo do diretório atual
    try:
        info["directories"]["current_dir_contents"] = os.listdir(".")
    except Exception as e:
        info["directories"]["current_dir_contents"] = f"Error: {str(e)}"
        
    # Se estamos no Render, verificar o diretório de projeto
    if os.path.exists("/opt/render/project/src"):
        try:
            info["directories"]["/opt/render/project/src_contents"] = os.listdir("/opt/render/project/src")
        except Exception as e:
            info["directories"]["/opt/render/project/src_contents"] = f"Error: {str(e)}"
    
    return info

if __name__ == "__main__":
    info = debug_info()
    print(json.dumps(info, indent=2))
    
    # Salvar em um arquivo para referência
    with open("debug_paths_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print("\nInformações de depuração salvas em debug_paths_info.json")
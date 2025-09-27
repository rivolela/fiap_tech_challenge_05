#!/bin/bash
# debug_render.sh - Script para diagnosticar problemas no ambiente Render

# Verificar ambiente
echo "=== Ambiente ==="
echo "Usuário atual: $(whoami)"
echo "Diretório atual: $(pwd)"
echo "Variáveis de ambiente:"
printenv | grep -E 'PYTHON|PORT|LOG|RENDER|PATH'

# Verificar diretórios
echo -e "\n=== Verificando diretórios importantes ==="
dirs=(
  "/opt/render/project"
  "/opt/render/project/src"
  "/opt/render/project/logs"
  "/opt/render/project/src/logs"
  "logs"
  "data"
  "data/logs"
  "data/processed"
  "models"
)

for dir in "${dirs[@]}"; do
  if [ -d "$dir" ]; then
    echo "✅ $dir existe"
    ls -la "$dir" | head -n 5
  else
    echo "❌ $dir não existe"
  fi
done

# Verificar arquivos críticos
echo -e "\n=== Verificando arquivos críticos ==="
files=(
  "models/scoring_model.pkl"
  "/opt/render/project/src/models/scoring_model.pkl"
  "data/processed/complete_processed_data.csv"
  "/opt/render/project/src/data/processed/complete_processed_data.csv"
  "src/api/scoring_api.py"
  "requirements.txt"
  "api_logs.log"
  "/opt/render/project/logs/api_logs.log"
)

for file in "${files[@]}"; do
  if [ -f "$file" ]; then
    echo "✅ $file existe ($(du -h "$file" | cut -f1))"
  else
    echo "❌ $file não existe"
  fi
done

# Verificar instalações de pacotes
echo -e "\n=== Verificando pacotes Python ==="
pip freeze | grep -E 'scikit-learn|pandas|fastapi|gunicorn|uvicorn'

# Tentar importar módulos críticos
echo -e "\n=== Verificando imports Python ==="
python3 -c "
import sys
print(f'Python version: {sys.version}')
print(f'Path: {sys.path}')
try:
    import pandas as pd
    print('✅ pandas OK')
except Exception as e:
    print(f'❌ pandas error: {e}')
try:
    import sklearn
    print(f'✅ scikit-learn OK (version {sklearn.__version__})')
except Exception as e:
    print(f'❌ scikit-learn error: {e}')
try:
    import fastapi
    print(f'✅ fastapi OK (version {fastapi.__version__})')
except Exception as e:
    print(f'❌ fastapi error: {e}')
try:
    import pickle
    print('✅ pickle OK')
except Exception as e:
    print(f'❌ pickle error: {e}')
"

# Testar se consegue carregar o modelo
echo -e "\n=== Tentando carregar o modelo ==="
python3 -c "
import pickle
import os

# Verificar vários caminhos possíveis para o modelo
possible_paths = [
    'models/scoring_model.pkl',
    './models/scoring_model.pkl',
    '../models/scoring_model.pkl',
    '/opt/render/project/src/models/scoring_model.pkl'
]

model_loaded = False
for path in possible_paths:
    if os.path.exists(path):
        print(f'Tentando carregar modelo de {path}...')
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f'✅ Modelo carregado com sucesso de {path}')
            model_loaded = True
            print(f'Tipo do modelo: {type(model)}')
            break
        except Exception as e:
            print(f'❌ Erro ao carregar de {path}: {e}')
    else:
        print(f'❌ Arquivo {path} não existe')

if not model_loaded:
    print('❌ Não foi possível carregar o modelo de nenhum caminho')
"

echo -e "\n=== Verificando logs de erro ==="
find /opt/render/project -name "*.log" -type f | xargs grep -i "error\|exception\|failed" | tail -n 30

echo -e "\n=== Fim do diagnóstico ==="
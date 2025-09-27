#!/bin/bash
# Script para testar se as alterações no random_seed estão funcionando
# Executa o treinamento do modelo várias vezes para verificar se as métricas são diferentes

echo "==================================================="
echo "Testando train_simple.py com seeds aleatórios"
echo "==================================================="

cd "$(dirname "$0")/.."

# Executar 3 vezes com seed aleatório (gerado automaticamente)
echo -e "\n\n===== EXECUÇÃO 1 - Seed aleatório ====="
python src/models/train_simple.py --model RandomForest --no-cv

echo -e "\n\n===== EXECUÇÃO 2 - Seed aleatório ====="
python src/models/train_simple.py --model RandomForest --no-cv

echo -e "\n\n===== EXECUÇÃO 3 - Seed aleatório ====="
python src/models/train_simple.py --model RandomForest --no-cv

# Executar 2 vezes com o mesmo seed fixo para garantir reprodutibilidade
echo -e "\n\n===== EXECUÇÃO 4 - Seed fixo (42) ====="
python src/models/train_simple.py --model RandomForest --no-cv --random-seed 42

echo -e "\n\n===== EXECUÇÃO 5 - Seed fixo (42) ====="
python src/models/train_simple.py --model RandomForest --no-cv --random-seed 42

echo -e "\n\n===== EXECUÇÃO 6 - Seed fixo diferente (123) ====="
python src/models/train_simple.py --model RandomForest --no-cv --random-seed 123

echo -e "\n\nVerifique no MLflow UI se as métricas são diferentes entre as execuções 1-3"
echo "e se são iguais entre as execuções 4-5 (mesmo seed)"
echo "Para visualizar: mlflow ui --port 5001"
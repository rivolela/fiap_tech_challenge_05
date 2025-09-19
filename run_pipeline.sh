#!/bin/bash
# Script para executar o pipeline completo de treinamento com MLflow
# Autor: GitHub Copilot
# Data: Setembro 2023

echo "=== PIPELINE DE TREINAMENTO COM MLFLOW ==="
echo "Verificando ambiente..."

# Ativar ambiente virtual se existir
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Ambiente virtual ativado"
else
    echo "⚠️ Ambiente virtual não encontrado. Executando setup..."
    bash setup.sh
    source .venv/bin/activate
fi

# Verificar dependências
echo "🔍 Verificando dependências..."
python scripts/check_env.py

# Verificar se MLflow está instalado
if python -c "import mlflow" &>/dev/null; then
    echo "✅ MLflow está instalado"
else
    echo "⚠️ MLflow não encontrado. Instalando..."
    pip install mlflow
fi

# Criar diretórios necessários
mkdir -p models data/insights data/visualizations

# Opções de execução
COMPARE=false
PORT=5001
START_SERVER=true

# Processar argumentos
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --compare) COMPARE=true ;;
        --port) PORT="$2"; shift ;;
        --no-server) START_SERVER=false ;;
        *) echo "Opção desconhecida: $1"; exit 1 ;;
    esac
    shift
done

# Configuramos o MLflow para usar armazenamento local
echo "� Configurando MLflow para usar armazenamento local..."
export MLFLOW_TRACKING_URI="file:./mlruns"
echo "✅ MLflow configurado para armazenar experimentos em ./mlruns"
echo ""

# Executar treinamento do modelo
echo "🔄 Iniciando treinamento do modelo..."
if [ "$COMPARE" = true ]; then
    python src/models/train_simple.py --compare
else
    python src/models/train_simple.py
fi

# Exibir mensagem final
echo ""
echo "🎉 Pipeline concluído!"
echo ""
echo "Para visualizar os resultados no MLflow, execute:"
echo "mlflow ui --port 5001"
echo ""
echo "E depois acesse http://localhost:5001 em seu navegador"
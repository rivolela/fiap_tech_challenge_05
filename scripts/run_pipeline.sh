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
    bash scripts/setup.sh
    source .venv/bin/activate
fi

# Verificar dependências
echo "🔍 Verificando dependências..."
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, mlflow" 2>/dev/null || {
    echo "⚠️ Dependências não encontradas. Instalando..."
    pip install -r requirements.txt
}

# Verificar se o diretório data/raw existe
if [ ! -d "data/raw" ]; then
    echo "❌ Diretório data/raw não encontrado"
    echo "O diretório data/raw deve conter os arquivos JSON de entrada"
    exit 1
fi

# Diretórios de saída
PROCESSED_DIR="data/processed"
SPLITS_DIR="${PROCESSED_DIR}/splits"
INSIGHTS_DIR="data/insights"
MODELS_DIR="models"

# Criar diretórios se não existirem
mkdir -p ${PROCESSED_DIR} ${SPLITS_DIR} ${INSIGHTS_DIR} ${MODELS_DIR} 

echo "✅ Ambiente configurado"
echo

# Variáveis de controle (padrões)
START_MLFLOW_SERVER=true
DO_CV=true
DO_COMPARE=false
PREVENT_LEAKAGE=true
DO_FEATURE_SELECTION=true
CV_FOLDS=5
MLFLOW_PORT=5001

# Processamento de argumentos
for arg in "$@"
do
    case $arg in
        --no-server)
            START_MLFLOW_SERVER=false
            shift
            ;;
        --no-cv)
            DO_CV=false
            shift
            ;;
        --compare)
            DO_COMPARE=true
            shift
            ;;
        --no-leakage-prevention)
            PREVENT_LEAKAGE=false
            shift
            ;;
        --no-feature-selection)
            DO_FEATURE_SELECTION=false
            shift
            ;;
        --cv-folds=*)
            CV_FOLDS="${arg#*=}"
            shift
            ;;
        --port=*)
            MLFLOW_PORT="${arg#*=}"
            shift
            ;;
        *)
            # Ignorar argumentos desconhecidos
            shift
            ;;
    esac
done

# Iniciar servidor MLflow se necessário
if $START_MLFLOW_SERVER; then
    echo "🚀 Iniciando servidor MLflow na porta ${MLFLOW_PORT}..."
    # Rodar em segundo plano
    mlflow ui --port ${MLFLOW_PORT} &
    MLFLOW_PID=$!
    echo "✅ Servidor MLflow iniciado (PID: ${MLFLOW_PID})"
    echo "📊 Acesse o MLflow UI em http://localhost:${MLFLOW_PORT}"
    echo
    # Garantir que o servidor termine quando o script terminar
    trap "echo '🛑 Encerrando servidor MLflow'; kill ${MLFLOW_PID} 2>/dev/null" EXIT
else
    echo "ℹ️ Modo sem servidor MLflow"
fi

# Construir argumentos para o script Python
TRAIN_ARGS=""
if [ "$DO_CV" = false ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --no-cv"
fi
if [ "$DO_COMPARE" = true ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --compare"
fi
if [ "$PREVENT_LEAKAGE" = false ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --no-leakage-prevention"
fi
if [ "$DO_FEATURE_SELECTION" = false ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --no-feature-selection"
fi
TRAIN_ARGS="${TRAIN_ARGS} --cv-folds ${CV_FOLDS}"

echo "🧪 Executando pipeline de treinamento com configurações:"
echo "   - Cross-validation: $([ "$DO_CV" = true ] && echo "Ativado (${CV_FOLDS} folds)" || echo "Desativado")"
echo "   - Comparação de modelos: $([ "$DO_COMPARE" = true ] && echo "Ativado" || echo "Desativado")"
echo "   - Prevenção de data leakage: $([ "$PREVENT_LEAKAGE" = true ] && echo "Ativado" || echo "Desativado")"
echo "   - Seleção de features: $([ "$DO_FEATURE_SELECTION" = true ] && echo "Ativado" || echo "Desativado")"
echo

# Executar o script Python de treinamento
echo "🔬 Executando treinamento do modelo..."
if python src/models/train_simple.py $TRAIN_ARGS; then
    echo "✅ Treinamento concluído com sucesso"
else
    echo "❌ Erro no treinamento do modelo"
    exit 1
fi

echo
echo "🏆 Pipeline concluído com sucesso!"
echo "📊 Modelo salvo em models/scoring_model.pkl"
echo

# Se iniciou o servidor MLflow, mantém ele rodando até o usuário encerrar
if $START_MLFLOW_SERVER; then
    echo "💡 Servidor MLflow continua em execução. Pressione Ctrl+C para encerrar."
    # Aguardar até o usuário pressionar Ctrl+C
    # Isso permite que o usuário continue vendo o MLflow UI
    wait ${MLFLOW_PID}
fi
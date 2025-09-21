#!/bin/bash
# Wrapper para manter compatibilidade após mover scripts para pasta scripts/
echo "⚙️ Redirecionando para scripts/run_pipeline.sh..."
./scripts/run_pipeline.sh "$@"

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

# Opções adicionais para prevenção de data leakage e validação cruzada
USE_CV=true
PREVENT_LEAKAGE=true
FEATURE_SELECTION=true
CV_FOLDS=5

# Processar argumentos adicionais
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --no-cv) USE_CV=false ;;
        --no-leakage-prevention) PREVENT_LEAKAGE=false ;;
        --no-feature-selection) FEATURE_SELECTION=false ;;
        --cv-folds) CV_FOLDS="$2"; shift ;;
    esac
    shift
done

# Construir o comando com base nas opções
CMD="python src/models/train_simple.py"

# Adicionar opções baseadas nas configurações
if [ "$COMPARE" = true ]; then
    CMD="$CMD --compare"
fi

if [ "$USE_CV" = false ]; then
    CMD="$CMD --no-cv"
fi

if [ "$PREVENT_LEAKAGE" = false ]; then
    CMD="$CMD --no-leakage-prevention"
fi

if [ "$FEATURE_SELECTION" = false ]; then
    CMD="$CMD --no-feature-selection"
fi

CMD="$CMD --cv-folds $CV_FOLDS"

# Executar o comando
echo "Executando: $CMD"
$CMD

# Exibir mensagem final
echo ""
echo "🎉 Pipeline concluído!"
echo ""
echo "Para visualizar os resultados no MLflow, execute:"
echo "mlflow ui --port 5001"
echo ""
echo "E depois acesse http://localhost:5001 em seu navegador"
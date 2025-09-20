#!/bin/bash
# Script para iniciar a API do modelo de scoring

set -e

echo "=== DECISION SCORING API ==="

# Verificar se o modelo existe
if [ ! -f "models/scoring_model.pkl" ]; then
    echo "❌ Modelo não encontrado em models/scoring_model.pkl"
    echo "Execute o treinamento primeiro com: ./run_pipeline.sh"
    exit 1
fi

# Verificar se as dependências estão instaladas
pip show fastapi uvicorn > /dev/null || {
    echo "📦 Instalando dependências necessárias..."
    pip install fastapi uvicorn
}

# Definir variáveis de ambiente
export PYTHONPATH=$(pwd)

# Iniciar o servidor API
echo "🚀 Iniciando API no endereço http://localhost:8000"
echo "📚 Documentação disponível em http://localhost:8000/docs"
echo ""
echo "📌 API Keys disponíveis para teste:"
echo "   - your-api-key (admin)"
echo "   - test-api-key (read-only)"
echo ""
echo "📌 Para autenticar, você pode usar:"
echo "   - Parâmetro de consulta: ?api_key=your-api-key"
echo "   - Cabeçalho HTTP: X-API-Key: your-api-key"
echo ""
echo "Pressione Ctrl+C para encerrar o servidor"

# Verificar se a porta 8000 já está em uso
PORT=8000
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo ""
    echo "⚠️  Atenção: A porta $PORT já está em uso!"
    echo "Isso geralmente ocorre quando:"
    echo "  1. Uma instância anterior da API ainda está em execução"
    echo "  2. Outro serviço está usando a porta $PORT"
    echo ""
    echo "Escolha uma opção:"
    echo "  1) Encerrar o processo que está usando a porta $PORT e continuar"
    echo "  2) Usar uma porta diferente"
    echo "  3) Cancelar"
    read -p "Opção [1-3]: " option
    
    case $option in
        1)
            echo "Encerrando o processo que está usando a porta $PORT..."
            # Em macOS (detectamos pelo comando)
            if command -v lsof &> /dev/null; then
                lsof -ti :$PORT | xargs kill -9
            # Em Linux
            else
                fuser -k ${PORT}/tcp
            fi
            echo "✅ Processo encerrado"
            ;;
        2)
            read -p "Digite o número da porta a ser usada: " PORT
            echo "✅ Usando porta $PORT"
            ;;
        3)
            echo "Operação cancelada"
            exit 0
            ;;
        *)
            echo "Opção inválida, saindo..."
            exit 1
            ;;
    esac
fi

# Iniciar o servidor
echo "Iniciando API na porta $PORT..."
uvicorn src.api.scoring_api:app --host 0.0.0.0 --port $PORT --reload
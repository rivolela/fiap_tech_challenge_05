#!/bin/bash
# Script para visualizar o MLflow UI após executar os testes

echo "==================================================="
echo "Visualização do MLflow UI"
echo "==================================================="

# Verifique se o MLflow está rodando
if ! pgrep -f "mlflow ui" > /dev/null; then
    echo "Iniciando MLflow UI na porta 5001..."
    mlflow ui --port 5001 &
    sleep 3  # Dar tempo para iniciar
    echo "MLflow UI iniciado em segundo plano."
fi

# Instruções para o usuário
echo -e "\n\nPara verificar se as métricas estão variando entre os treinamentos:"
echo "1. Abra http://localhost:5001 em seu navegador"
echo "2. Clique no experimento 'Decision-Scoring-Model'"
echo "3. Observe que cada execução agora tem um nome único com timestamp"
echo "4. Compare as métricas entre diferentes execuções"
echo "   - Execuções com seeds diferentes devem ter métricas diferentes"
echo "   - Execuções com o mesmo seed devem ter métricas idênticas"
echo -e "\nQuando terminar, pressione CTRL+C para encerrar o MLflow UI."

# Aguardar o usuário
echo -e "\nMLflow UI está rodando em http://localhost:5001"
echo "Pressione CTRL+C para encerrar..."

# Aguardar sinal para terminar
trap "echo -e '\nEncerrando MLflow UI...'; pkill -f 'mlflow ui'; echo 'Encerrado.'; exit 0" INT
while true; do sleep 1; done
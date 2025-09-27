#!/bin/bash
# Script para testar diferentes thresholds de classificação
# Este script ajusta o threshold via variável de ambiente e envia requisições de teste

echo "==================================================="
echo "Testando diferentes thresholds de classificação"
echo "==================================================="

cd "$(dirname "$0")/.."

# Definir dados de teste
TEST_DATA='{"prospect_id": 12345, "nome": "João Silva", "idade": 35, "experiencia": 8, "educacao": "Superior Completo", "area_formacao": "Engenharia", "anos_estudo": 16, "cargo_anterior": "Analista de Sistemas", "tempo_desempregado": 3, "salario_pretendido": 7500, "habilidades": ["Python", "SQL", "AWS"]}'

# Função para testar um threshold específico
test_threshold() {
    local threshold=$1
    echo -e "\n\n===== TESTANDO COM THRESHOLD = $threshold ====="
    export CLASSIFICATION_THRESHOLD=$threshold
    
    # Reiniciar a API com o novo threshold (se estiver rodando localmente)
    if pgrep -f "uvicorn src.api.scoring_api:app" > /dev/null; then
        echo "Reiniciando API local com novo threshold..."
        pkill -f "uvicorn src.api.scoring_api:app"
        nohup uvicorn src.api.scoring_api:app --reload > api.log 2>&1 &
        sleep 2
    fi
    
    # Enviar requisição
    echo "Enviando requisição de teste..."
    curl -s -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -H "x-api-key: teste123" \
         -d "$TEST_DATA" | jq .
}

# Testar uma série de thresholds
echo "Testando vários thresholds para avaliar o impacto nas predições..."

test_threshold 0.5   # Threshold padrão
test_threshold 0.3   # Threshold mais baixo
test_threshold 0.2   # Threshold ainda mais baixo
test_threshold 0.1   # Threshold muito baixo

echo -e "\n\nAnálise de thresholds concluída!"
echo "Um threshold mais baixo (como 0.1-0.3) deve aumentar o número de candidatos recomendados,"
echo "melhorando o recall às custas de possivelmente reduzir um pouco a precisão."
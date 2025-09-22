#!/bin/bash
# Script para iniciar a API do modelo de scoring com ambiente virtual
# Resolve o erro "No module named 'numpy._core'"

set -e

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== DECISION SCORING API ===${NC}"

# Verificar se o modelo existe
if [ ! -f "models/scoring_model.pkl" ]; then
    echo -e "${YELLOW}❌ Modelo não encontrado em models/scoring_model.pkl${NC}"
    echo -e "${YELLOW}Execute o treinamento primeiro com: ./scripts/run_pipeline.sh${NC}"
    exit 1
fi

# Verificar se o arquivo .env existe, se não criar a partir do exemplo
if [ ! -f .env ]; then
    echo -e "${YELLOW}Arquivo .env não encontrado, criando a partir do exemplo...${NC}"
    cp .env.example .env
    echo -e "${GREEN}Criado arquivo .env. Ajuste as configurações conforme necessário.${NC}"
fi

# Verificar e criar ambiente virtual Python
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Criando ambiente virtual Python (.venv)...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}Ambiente virtual criado em .venv${NC}"
fi

# Ativar ambiente virtual
echo -e "${YELLOW}Ativando ambiente virtual...${NC}"
source .venv/bin/activate

# Atualizar pip
echo -e "${YELLOW}Atualizando pip...${NC}"
pip install --upgrade pip

# Instalar NumPy em versão compatível primeiro
echo -e "${YELLOW}Instalando NumPy em versão compatível (1.24.3)...${NC}"
pip install numpy==1.24.3

# Instalar demais dependências do requirements.txt
echo -e "${YELLOW}Instalando demais dependências...${NC}"
pip install -r requirements.txt

# Verificar instalação de NumPy e scikit-learn
echo -e "${YELLOW}Verificando instalação:${NC}"
pip list | grep numpy
pip list | grep scikit-learn

echo -e "${GREEN}Ambiente configurado com sucesso!${NC}"

# Executar a API
echo -e "${YELLOW}🚀 Iniciando a API na porta 8000...${NC}"
python -m uvicorn src.api.scoring_api:app --host 0.0.0.0 --port 8000 --reload

# Notas de uso:
# Para iniciar com recarga automática (desenvolvimento):
#   uvicorn src.api.scoring_api:app --host 0.0.0.0 --port 8000 --reload
# 
# Para iniciar com múltiplos workers em produção:
#   gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 src.api.scoring_api:app
#
# Documentação da API disponível em:
#   http://localhost:8000/docs
#
# Uso básico da API:
#   curl -X POST "http://localhost:8000/predict/" \
#        -H "X-API-Key: your-api-key" \
#        -H "Content-Type: application/json" \
#        -d '{
#              "idade": 30,
#              "experiencia": 5,
#              "educacao": "ensino_superior",
#              "area_formacao": "tecnologia",
#              "habilidades": ["python", "machine_learning"],
#              "vaga_titulo": "Desenvolvedor Python",
#              "vaga_area": "tecnologia",
#              "vaga_senioridade": "pleno"
#            }'
#
# Healthcheck:
#   curl "http://localhost:8000/health"

# Implementação de recursos:
# ✅ Autenticação via API key
# ✅ Validação de dados de entrada
# ✅ Documentação interativa (Swagger)
# ✅ Endpoints para predição individual e em lote
# ✅ Rota de verificação de saúde
# ✅ Métricas de performance
# ✅ Logging de requisições
# ✅ Tratamento de erros com mensagens descritivas
# ✅ Inferência de valores faltantes
# ✅ Geração de comentários via LLM
#
# O sistema também é compatível com Docker, sendo possível
# executar a API com:
#   docker build -t decision-scoring-api .
#   docker run -p 8000:8000 decision-scoring-api
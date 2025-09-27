#!/bin/bash
# Script para retreinar o modelo com as versões atuais das bibliotecas
# Resolve o erro "No module named 'numpy._core'"

set -e

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== RETREINAMENTO DO MODELO SCORING ===${NC}"
echo -e "${YELLOW}Este script vai retreinar o modelo com as versões atuais do NumPy e scikit-learn${NC}"

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
echo -e "${YELLOW}Instalando NumPy em versão compatível com Python 3.12...${NC}"
pip install --only-binary=:all: numpy==1.26.0

# Instalar scikit-learn específica
echo -e "${YELLOW}Instalando scikit-learn específica...${NC}"
pip install --only-binary=:all: scikit-learn==1.7.1

# Instalar demais dependências do requirements.txt
echo -e "${YELLOW}Instalando demais dependências...${NC}"
pip install -r requirements.txt

# Verificar instalação
echo -e "${YELLOW}Verificando instalação:${NC}"
pip list | grep numpy
pip list | grep scikit-learn

echo -e "${GREEN}Ambiente configurado com sucesso!${NC}"

# Executar o treinamento do modelo
echo -e "${YELLOW}Executando o pipeline de treinamento...${NC}"
python -m src.models.train_model

# Verificar se o modelo foi gerado corretamente
if [ -f "models/scoring_model.pkl" ]; then
    echo -e "${GREEN}✅ Modelo retreinado com sucesso em models/scoring_model.pkl${NC}"
else
    echo -e "${YELLOW}❌ Falha ao retreinar o modelo${NC}"
    exit 1
fi

echo -e "${YELLOW}Agora você pode executar a API com o novo modelo usando:${NC}"
echo -e "${GREEN}./scripts/start_api.sh${NC}"
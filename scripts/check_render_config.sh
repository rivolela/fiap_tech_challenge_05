#!/bin/bash

# Script para verificar se a configuração local é compatível com o Render

echo "🔍 Verificando configuração para deploy no Render..."

# Cores para melhor visualização
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Verificar se existe o arquivo .env
echo -e "\n${YELLOW}1. Verificando arquivo .env${NC}"
if [ -f .env ]; then
    echo -e "${GREEN}✓ Arquivo .env encontrado${NC}"
    source .env
else
    echo -e "${RED}✗ Arquivo .env não encontrado${NC}"
    echo -e "  Criando .env a partir do exemplo..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Arquivo .env criado a partir do exemplo${NC}"
        source .env
    else
        echo -e "${RED}✗ Arquivo .env.example não encontrado${NC}"
        echo -e "  Verifique sua instalação"
        exit 1
    fi
fi

# 2. Verificar estrutura de arquivos críticos
echo -e "\n${YELLOW}2. Verificando arquivos críticos para o Render${NC}"

REQUIRED_FILES=(
    "requirements.txt"
    "Dockerfile"
    "src/api/scoring_api.py"
    "src/api/model_loader.py"
    "models/scoring_model.pkl"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $file encontrado${NC}"
    else
        echo -e "${RED}✗ $file não encontrado${NC}"
        echo -e "  Este arquivo é necessário para o deploy no Render"
    fi
done

# 3. Verificar configurações do PYTHONPATH
echo -e "\n${YELLOW}3. Verificando configurações de PYTHONPATH${NC}"
if [ "$PYTHONPATH" == "/opt/render/project/src" ]; then
    echo -e "${GREEN}✓ PYTHONPATH configurado corretamente para o Render${NC}"
else
    echo -e "${RED}✗ PYTHONPATH não está configurado para o Render${NC}"
    echo -e "  Valor atual: $PYTHONPATH"
    echo -e "  Valor esperado: /opt/render/project/src"
fi

# 4. Verificar configurações de segurança
echo -e "\n${YELLOW}4. Verificando configurações de segurança${NC}"
if [ "$API_KEY_SALT" == "salt-secreto-mude-em-producao" ]; then
    echo -e "${RED}✗ API_KEY_SALT está usando valor padrão${NC}"
    echo -e "  Mude este valor antes de fazer deploy em produção"
else
    echo -e "${GREEN}✓ API_KEY_SALT personalizado${NC}"
fi

# 5. Verificar compatibilidade do Docker
echo -e "\n${YELLOW}5. Verificando compatibilidade do Docker${NC}"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker encontrado${NC}"
    
    # Verificar se o Dockerfile está configurado corretamente
    if grep -q "WORKDIR /opt/render/project/src" Dockerfile; then
        echo -e "${GREEN}✓ Dockerfile configurado para o Render${NC}"
    else
        echo -e "${RED}✗ Dockerfile não está configurado para o Render${NC}"
        echo -e "  WORKDIR deve ser /opt/render/project/src"
    fi
else
    echo -e "${YELLOW}⚠ Docker não encontrado${NC}"
    echo -e "  Instale o Docker para testar localmente a configuração"
fi

# 6. Verificar o MLflow model
echo -e "\n${YELLOW}6. Verificando modelo de ML${NC}"
if [ -f "models/scoring_model.pkl" ]; then
    echo -e "${GREEN}✓ Modelo ML encontrado${NC}"
    
    # Verificar tamanho do modelo
    MODEL_SIZE=$(du -h "models/scoring_model.pkl" | cut -f1)
    echo -e "  Tamanho do modelo: $MODEL_SIZE"
else
    echo -e "${RED}✗ Modelo ML não encontrado em models/scoring_model.pkl${NC}"
    echo -e "  Verifique se o modelo foi treinado e exportado corretamente"
fi

# 7. Verificar arquivos de monitoramento
echo -e "\n${YELLOW}7. Verificando arquivos de monitoramento${NC}"
if [ -f "src/monitoring/drift_detector.py" ] && [ -f "src/dashboard/dashboard.py" ]; then
    echo -e "${GREEN}✓ Sistema de monitoramento encontrado${NC}"
else
    echo -e "${RED}✗ Sistema de monitoramento não encontrado ou incompleto${NC}"
    echo -e "  Verifique se os arquivos de monitoramento foram criados"
fi

echo -e "\n${GREEN}=== Verificação de configuração para o Render concluída ===${NC}"
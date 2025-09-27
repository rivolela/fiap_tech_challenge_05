#!/bin/bash

# Script simplificado de CI/CD para deploy em servidores como Render
# Este script executa os testes e, se bem-sucedidos, faz o deploy da aplicação

# Definindo cores para saída
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Definindo diretório raiz do projeto
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

echo -e "${BLUE}===== PIPELINE DE DEPLOY INICIADA =====${NC}"
echo -e "${YELLOW}Diretório do projeto:${NC} $PROJECT_ROOT"

# Verificando variáveis de ambiente necessárias
if [ -z "$APP_ENV" ]; then
    APP_ENV="development"
fi

echo -e "${YELLOW}Ambiente:${NC} $APP_ENV"

# Executando os testes
echo -e "\n${BLUE}===== EXECUTANDO TESTES =====${NC}"
if [ -f "$PROJECT_ROOT/scripts/run_tests.sh" ]; then
    bash "$PROJECT_ROOT/scripts/run_tests.sh"
    TEST_RESULT=$?
    
    if [ $TEST_RESULT -ne 0 ]; then
        echo -e "\n${RED}ERRO: Os testes falharam. O deploy foi cancelado.${NC}"
        exit $TEST_RESULT
    else
        echo -e "\n${GREEN}✓ Todos os testes passaram!${NC}"
    fi
else
    echo -e "${YELLOW}AVISO: O script de testes não foi encontrado em scripts/run_tests.sh${NC}"
    echo -e "${YELLOW}Continuando sem executar testes...${NC}"
fi

# Aplicando migrações de banco de dados (se houver)
echo -e "\n${BLUE}===== APLICANDO MIGRAÇÕES =====${NC}"
# Aqui você pode adicionar comandos específicos para migrações de banco de dados
# Exemplo para Django:
# python manage.py migrate

# Verificando pré-requisitos para o ambiente de produção
if [ "$APP_ENV" == "production" ]; then
    echo -e "\n${BLUE}===== VERIFICAÇÕES DE PRÉ-REQUISITOS =====${NC}"
    
    # Verificar se a configuração de produção está completa
    if [ ! -f "$PROJECT_ROOT/config/render/.env.render" ]; then
        echo -e "${RED}ERRO: Arquivo de configuração de produção não encontrado.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Configurações de produção verificadas.${NC}"
fi

# Iniciando a aplicação
echo -e "\n${BLUE}===== INICIANDO APLICAÇÃO =====${NC}"

if [ "$APP_ENV" == "production" ]; then
    # Comando para iniciar em produção
    if [ -f "$PROJECT_ROOT/config/render/Procfile" ]; then
        echo -e "${YELLOW}Iniciando com Procfile...${NC}"
        # Na produção, o Render geralmente gerencia o processo usando o Procfile
    else
        echo -e "${YELLOW}Iniciando servidor de produção...${NC}"
        # Comando alternativo para iniciar o servidor
        # gunicorn src.api.scoring_api:app --bind 0.0.0.0:$PORT
    fi
else
    # Comando para iniciar em desenvolvimento/staging
    echo -e "${YELLOW}Iniciando servidor de desenvolvimento...${NC}"
    # python -m uvicorn src.api.scoring_api:app --reload --port 8000
fi

echo -e "\n${BLUE}===== PIPELINE DE DEPLOY CONCLUÍDA COM SUCESSO =====${NC}"
exit 0
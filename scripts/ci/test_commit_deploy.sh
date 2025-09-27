#!/bin/bash

# Script de CI/CD para testes, commit e deploy
# Uso: ./test_commit_deploy.sh ["Mensagem do commit"]
# Se não for fornecida mensagem, será gerada automaticamente com LLM

# Definindo cores para saída
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Definindo diretório raiz do projeto
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Função para gerar mensagem de commit usando OpenAI API
generate_commit_message() {
    # Verificar se o arquivo .env existe com a chave da API
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        echo -e "${YELLOW}AVISO: Arquivo .env não encontrado. Não é possível gerar mensagem automaticamente.${NC}"
        return 1
    fi

    # Tentar carregar a chave da API do arquivo .env
    if grep -q "OPENAI_API_KEY" "$PROJECT_ROOT/.env"; then
        source "$PROJECT_ROOT/.env"
    else
        echo -e "${YELLOW}AVISO: OPENAI_API_KEY não encontrada em .env. Não é possível gerar mensagem automaticamente.${NC}"
        return 1
    fi

    # Obter as alterações desde o último commit
    DIFF_SUMMARY=$(git diff --staged --stat | cat)
    FILES_CHANGED=$(git diff --staged --name-only | cat)
    
    # Se não houver alterações staged, usar todas as alterações
    if [ -z "$DIFF_SUMMARY" ]; then
        DIFF_SUMMARY=$(git diff --stat | cat)
        FILES_CHANGED=$(git diff --name-only | cat)
    fi

    # Obter uma amostra do conteúdo alterado (limitado para não sobrecarregar a API)
    DIFF_CONTENT=$(git diff --staged | head -n 50)
    
    if [ -z "$DIFF_SUMMARY" ]; then
        echo -e "${YELLOW}AVISO: Nenhuma alteração detectada para análise.${NC}"
        return 1
    fi

    echo -e "${YELLOW}Gerando mensagem de commit com base nas alterações...${NC}"
    
    # Preparar o prompt para a API
    PROMPT="Por favor, gere uma mensagem de commit concisa e descritiva no formato do Conventional Commits para as seguintes alterações. Formato esperado: tipo(escopo opcional): descrição. Tipos possíveis: feat, fix, docs, style, refactor, test, chore.

Arquivos alterados:
$FILES_CHANGED

Resumo das alterações:
$DIFF_SUMMARY

Amostra do conteúdo:
$DIFF_CONTENT

Mensagem de commit:"
    
    # Chamar a API OpenAI
    API_RESPONSE=$(curl -s https://api.openai.com/v1/chat/completions \
        -H "Authorization: Bearer $OPENAI_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"gpt-3.5-turbo\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
            \"temperature\": 0.5,
            \"max_tokens\": 100
        }")

    # Extrair a mensagem gerada
    GENERATED_MSG=$(echo $API_RESPONSE | grep -o '"content":"[^"]*"' | cut -d'"' -f4)
    
    if [ -z "$GENERATED_MSG" ]; then
        echo -e "${RED}ERRO: Falha ao gerar mensagem de commit.${NC}"
        return 1
    else
        # Remover aspas e caracteres especiais
        GENERATED_MSG=$(echo "$GENERATED_MSG" | tr -d '\r\n')
        echo "$GENERATED_MSG"
        return 0
    fi
}

# Verificando se foi fornecida uma mensagem de commit
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Nenhuma mensagem de commit fornecida. Tentando gerar automaticamente...${NC}"
    GENERATED_MSG=$(generate_commit_message)
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Mensagem gerada: ${NC}\"$GENERATED_MSG\""
        read -p "Usar esta mensagem? (s/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Ss]$ ]]; then
            COMMIT_MESSAGE="$GENERATED_MSG"
        else
            echo -e "${RED}Por favor, forneça uma mensagem de commit manualmente.${NC}"
            echo "Uso: $0 \"Sua mensagem de commit aqui\""
            exit 1
        fi
    else
        echo -e "${RED}Por favor, forneça uma mensagem de commit como argumento.${NC}"
        echo "Uso: $0 \"Sua mensagem de commit aqui\""
        exit 1
    fi
else
    COMMIT_MESSAGE="$1"
fi
BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo -e "${BLUE}===== PIPELINE CI/CD INICIADA =====${NC}"
echo -e "${YELLOW}Diretório do projeto:${NC} $PROJECT_ROOT"
echo -e "${YELLOW}Branch atual:${NC} $BRANCH"

# Verificando se o repositório está limpo
if [[ $(git status --porcelain) ]]; then
    echo -e "${YELLOW}Existem alterações não commitadas no repositório.${NC}"
else
    echo -e "${YELLOW}Nenhuma alteração detectada para commit.${NC}"
    
    # Pergunta ao usuário se deseja continuar
    read -p "Continuar mesmo assim? (s/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        echo -e "${YELLOW}Operação cancelada pelo usuário.${NC}"
        exit 0
    fi
fi

# Executando os testes
echo -e "\n${BLUE}===== EXECUTANDO TESTES =====${NC}"
if [ -f "$PROJECT_ROOT/scripts/run_tests.sh" ]; then
    bash "$PROJECT_ROOT/scripts/run_tests.sh"
    TEST_RESULT=$?
    
    if [ $TEST_RESULT -ne 0 ]; then
        echo -e "\n${RED}ERRO: Os testes falharam. O processo de CI/CD foi interrompido.${NC}"
        
        # Pergunta ao usuário se deseja continuar mesmo com falhas
        read -p "Continuar mesmo com falhas nos testes? (s/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Ss]$ ]]; then
            echo -e "${YELLOW}Operação cancelada pelo usuário.${NC}"
            exit $TEST_RESULT
        fi
    else
        echo -e "\n${GREEN}✓ Todos os testes passaram!${NC}"
    fi
else
    echo -e "${YELLOW}AVISO: O script de testes não foi encontrado em scripts/run_tests.sh${NC}"
    
    # Pergunta ao usuário se deseja continuar sem executar testes
    read -p "Continuar sem executar os testes? (s/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        echo -e "${YELLOW}Operação cancelada pelo usuário.${NC}"
        exit 1
    fi
fi

# Fazendo commit das alterações
echo -e "\n${BLUE}===== REALIZANDO COMMIT =====${NC}"
git add .
git commit -m "$COMMIT_MESSAGE"
COMMIT_RESULT=$?

if [ $COMMIT_RESULT -ne 0 ]; then
    echo -e "\n${RED}ERRO: Falha ao criar commit.${NC}"
    exit $COMMIT_RESULT
else
    COMMIT_HASH=$(git rev-parse --short HEAD)
    echo -e "\n${GREEN}✓ Commit criado com sucesso: ${COMMIT_HASH} - ${COMMIT_MESSAGE}${NC}"
fi

# Realizando push
echo -e "\n${BLUE}===== REALIZANDO PUSH =====${NC}"
echo -e "${YELLOW}Enviando alterações para o branch:${NC} $BRANCH"

read -p "Confirmar push para '$BRANCH'? (s/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    echo -e "${YELLOW}Push cancelado pelo usuário.${NC}"
    exit 0
fi

git push origin "$BRANCH"
PUSH_RESULT=$?

if [ $PUSH_RESULT -ne 0 ]; then
    echo -e "\n${RED}ERRO: Falha ao fazer push para o repositório remoto.${NC}"
    exit $PUSH_RESULT
else
    echo -e "\n${GREEN}✓ Push realizado com sucesso para ${BRANCH}${NC}"
fi

# Notificando sobre o deploy
if [ "$BRANCH" == "main" ]; then
    echo -e "\n${BLUE}===== DEPLOY EM PRODUÇÃO =====${NC}"
    echo -e "${YELLOW}O push foi feito para o branch principal.${NC}"
    echo "Se o seu projeto estiver configurado com CI/CD, o deploy em produção será iniciado automaticamente."
    echo "Verifique o painel de CI/CD para acompanhar o progresso."
elif [ "$BRANCH" == "develop" ]; then
    echo -e "\n${BLUE}===== DEPLOY EM STAGING =====${NC}"
    echo -e "${YELLOW}O push foi feito para o branch de desenvolvimento.${NC}"
    echo "Se o seu projeto estiver configurado com CI/CD, o deploy em staging será iniciado automaticamente."
    echo "Verifique o painel de CI/CD para acompanhar o progresso."
else
    echo -e "\n${YELLOW}AVISO: O push foi feito para um branch que não aciona deploy automático.${NC}"
    echo "Para fazer deploy, faça merge deste branch com 'develop' ou 'main'."
fi

echo -e "\n${BLUE}===== PIPELINE CI/CD CONCLUÍDA COM SUCESSO =====${NC}"
exit 0
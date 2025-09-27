#!/bin/bash

# Script para gerar mensagens de commit automaticamente com base nas alterações
# usando modelos de linguagem (LLM)
#
# Uso: ./generate_commit_message.sh
#
# O script analisa as alterações no repositório e usa a API do OpenAI
# para gerar uma mensagem de commit seguindo o padrão Conventional Commits.

# Definindo cores para saída
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Definindo diretório raiz do projeto
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Verificar se há alterações no repositório
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo -e "${RED}ERRO: Não está em um repositório git.${NC}"
    exit 1
fi

# Verificar se o arquivo .env existe com a chave da API
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${RED}ERRO: Arquivo .env não encontrado na raiz do projeto.${NC}"
    echo "Crie um arquivo .env com a variável OPENAI_API_KEY definida."
    exit 1
fi

# Carregar a chave da API do arquivo .env
if grep -q "OPENAI_API_KEY" "$PROJECT_ROOT/.env"; then
    source "$PROJECT_ROOT/.env"
else
    echo -e "${RED}ERRO: OPENAI_API_KEY não encontrada em .env.${NC}"
    echo "Adicione a linha OPENAI_API_KEY=sua_chave_aqui ao arquivo .env"
    exit 1
fi

# Verificar se a chave da API foi carregada
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}ERRO: OPENAI_API_KEY está vazia.${NC}"
    exit 1
fi

# Verificar se há alterações staged para commit
STAGED_CHANGES=$(git diff --staged --name-only)

# Se não houver alterações staged, perguntar se quer incluir todas
if [ -z "$STAGED_CHANGES" ]; then
    echo -e "${YELLOW}Nenhuma alteração preparada para commit.${NC}"
    
    # Verificar se há alterações não staged
    UNSTAGED_CHANGES=$(git diff --name-only)
    
    if [ -z "$UNSTAGED_CHANGES" ]; then
        echo -e "${RED}ERRO: Nenhuma alteração detectada no repositório.${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Existem alterações não preparadas para commit:${NC}"
    git diff --name-only
    
    read -p "Deseja adicionar todas as alterações ao staging? (s/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        git add .
        echo -e "${GREEN}Alterações adicionadas ao staging.${NC}"
    else
        echo -e "${YELLOW}Usando alterações não staged para gerar a mensagem (mas você precisará adicionar manualmente).${NC}"
    fi
fi

echo -e "${BLUE}===== GERANDO MENSAGEM DE COMMIT =====${NC}"

# Obter as alterações
DIFF_SUMMARY=$(git diff --staged --stat)
FILES_CHANGED=$(git diff --staged --name-only)

# Se não houver alterações staged, usar todas as alterações
if [ -z "$DIFF_SUMMARY" ]; then
    DIFF_SUMMARY=$(git diff --stat)
    FILES_CHANGED=$(git diff --name-only)
fi

# Obter uma amostra do conteúdo alterado (limitado para não sobrecarregar a API)
DIFF_CONTENT=$(git diff --staged | head -n 100)
    
if [ -z "$DIFF_SUMMARY" ]; then
    echo -e "${RED}ERRO: Nenhuma alteração detectada para análise.${NC}"
    exit 1
fi

echo -e "${YELLOW}Analisando alterações...${NC}"
echo "Arquivos alterados:"
echo "$FILES_CHANGED"

# Preparar o prompt para a API
PROMPT="Por favor, gere uma mensagem de commit concisa e descritiva no formato do Conventional Commits para as seguintes alterações. Formato esperado: tipo(escopo opcional): descrição. Tipos possíveis: feat, fix, docs, style, refactor, test, chore.

Arquivos alterados:
$FILES_CHANGED

Resumo das alterações:
$DIFF_SUMMARY

Amostra do conteúdo:
$DIFF_CONTENT

Mensagem de commit:"

echo -e "\n${YELLOW}Consultando API do OpenAI...${NC}"

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
    echo "Resposta da API:"
    echo "$API_RESPONSE"
    exit 1
fi

# Remover aspas e caracteres especiais
GENERATED_MSG=$(echo "$GENERATED_MSG" | tr -d '\r\n')

echo -e "\n${GREEN}Mensagem de commit gerada:${NC}"
echo -e "\n\"${YELLOW}$GENERATED_MSG${NC}\""

# Perguntar se quer usar esta mensagem
read -p "Usar esta mensagem para commit? (s/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    # Verificar novamente se há alterações staged
    if [ -z "$(git diff --staged --name-only)" ]; then
        echo -e "${RED}AVISO: Nenhuma alteração preparada para commit.${NC}"
        read -p "Adicionar todas as alterações ao staging? (s/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Ss]$ ]]; then
            git add .
        else
            echo -e "${YELLOW}Operação cancelada. Use git add para preparar os arquivos e tente novamente.${NC}"
            exit 0
        fi
    fi
    
    # Realizar o commit
    git commit -m "$GENERATED_MSG"
    COMMIT_RESULT=$?
    
    if [ $COMMIT_RESULT -eq 0 ]; then
        echo -e "\n${GREEN}✓ Commit realizado com sucesso!${NC}"
    else
        echo -e "\n${RED}✗ Falha ao realizar commit.${NC}"
        exit $COMMIT_RESULT
    fi
else
    echo -e "${YELLOW}Mensagem copiada para a área de transferência (quando disponível).${NC}"
    echo "$GENERATED_MSG" | pbcopy 2>/dev/null || echo "$GENERATED_MSG" | xclip -selection clipboard 2>/dev/null || true
    echo -e "${YELLOW}Operação de commit cancelada. Você pode usar a mensagem gerada manualmente.${NC}"
fi

exit 0
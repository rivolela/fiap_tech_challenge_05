#!/bin/bash

# Script para facilitar a execução do processo de reorganização do projeto
# 
# Uso: ./reorganize_project.sh [--dry-run] [--backup]
#
# Opções:
#   --dry-run   Apenas mostra o que seria feito, sem fazer alterações
#   --backup    Cria um backup do projeto antes de fazer alterações

# Definindo diretório do script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# Definindo caminho para os scripts de reorganização
REORGANIZE_SCRIPT="$SCRIPT_DIR/reorganize_project.py"
VERIFY_SCRIPT="$SCRIPT_DIR/verify_reorganization.py"

echo "=== Script de Reorganização do Projeto Decision Scoring ==="
echo "Diretório do projeto: $PROJECT_ROOT"

# Verificando se os scripts Python existem
if [ ! -f "$REORGANIZE_SCRIPT" ]; then
    echo "ERRO: Script de reorganização não encontrado: $REORGANIZE_SCRIPT"
    exit 1
fi

if [ ! -f "$VERIFY_SCRIPT" ]; then
    echo "AVISO: Script de verificação não encontrado: $VERIFY_SCRIPT"
fi

# Executando script de reorganização
echo -e "\n=== Executando reorganização... ==="
python3 "$REORGANIZE_SCRIPT" "$@"

# Verificando resultado
RESULT=$?
if [ $RESULT -ne 0 ]; then
    echo "ERRO: Reorganização falhou com código $RESULT"
    exit $RESULT
fi

# Verificando se há script de verificação
if [ -f "$VERIFY_SCRIPT" ]; then
    # Perguntando ao usuário se deseja executar a verificação
    read -p "Deseja executar verificação após reorganização? (s/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        echo -e "\n=== Executando verificação... ==="
        python3 "$VERIFY_SCRIPT"
    fi
fi

echo -e "\n=== Processo de reorganização concluído ==="
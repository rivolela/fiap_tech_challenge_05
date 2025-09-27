#!/bin/bash
# Script para comitar as modificações relacionadas à correção de logs

# Adicionar os arquivos modificados
git add src/api/scoring_api.py
git add scripts/fix_logs.sh
git add scripts/start_api_local.sh
git add docs/log_fix.md

# Comitar as alterações
git commit -m "fix: Corrige problema com logs da API não atualizados

- Melhora a inicialização de logs com definição explícita de permissões
- Adiciona script de diagnóstico e correção (scripts/fix_logs.sh)
- Adiciona script simplificado para iniciar a API (scripts/start_api_local.sh)
- Adiciona documentação detalhada sobre o problema e solução (docs/log_fix.md)"

echo "✅ Alterações commitadas com sucesso!"
echo "Use 'git push' para enviar as alterações para o repositório remoto."
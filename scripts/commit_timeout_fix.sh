#!/bin/bash
# Script para criar um commit com todas as mudanças

echo "=== Criando commit com todas as mudanças ==="

# Adicionar todos os arquivos modificados
git add .

# Criar o commit com a mensagem fornecida
git commit -m "fix: Resolve problemas de worker timeout no Render"

# Adicionar detalhes ao commit
git commit --amend -m "fix: Resolve problemas de worker timeout no Render

Implementações:
- Aumenta timeout do Gunicorn para 120s
- Reduz número de workers para 2 para economizar memória
- Adiciona script check_memory.py para monitoramento de recursos
- Otimiza pipeline de treinamento com BackgroundTasks do FastAPI
- Configura variáveis de ambiente para limitar uso de memória
- Adiciona graceful-timeout e keep-alive para gerenciamento de conexões
- Cria .env.render e scripts/startup.sh para configurações específicas
- Documenta soluções em docs/resolucao_timeout_render.md"

echo "=== Commit criado com sucesso ==="
echo "Para enviar as mudanças para o repositório remoto, execute:"
echo "git push origin main"
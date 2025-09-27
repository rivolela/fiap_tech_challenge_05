# Scripts de Reorganização do Projeto

Este diretório contém scripts para reorganizar a estrutura do projeto Decision Scoring, movendo arquivos para locais mais apropriados e atualizando referências.

## Arquivos Disponíveis

- `reorganize_project.py` - Script principal que executa a reorganização
- `verify_reorganization.py` - Script para verificar se a reorganização foi bem-sucedida
- `reorganize_project.sh` - Shell script para facilitar a execução do processo

## Como Usar

### Simulação (Dry Run)

Para ver o que seria feito sem fazer alterações reais:

```bash
./reorganize_project.sh --dry-run
```

### Reorganização com Backup

Para executar a reorganização com um backup do estado atual:

```bash
./reorganize_project.sh --backup
```

### Reorganização Completa

Para executar a reorganização completa:

```bash
./reorganize_project.sh
```

## O que os Scripts Fazem

1. **reorganize_project.py**:
   - Cria a nova estrutura de diretórios
   - Move arquivos para seus novos locais
   - Remove arquivos temporários
   - Atualiza referências nos arquivos
   - Atualiza o README.md com a nova estrutura
   - Cria um arquivo CONTRIBUTING.md

2. **verify_reorganization.py**:
   - Verifica se a estrutura de diretórios foi criada corretamente
   - Verifica se os arquivos foram movidos para os locais corretos
   - Verifica se os imports do projeto ainda funcionam
   - Executa os testes do projeto para garantir que tudo ainda funciona

3. **reorganize_project.sh**:
   - Script de conveniência para executar os dois scripts acima

## Plano de Movimentação de Arquivos

| Arquivo Original | Novo Local |
|------------------|------------|
| config/docker/api/Dockerfile | config/docker/api/config/docker/api/Dockerfile |
| config/docker/api/config/docker/dashboard/Dockerfile | config/docker/dashboard/config/docker/api/Dockerfile |
| config/docker/docker-compose.yml | config/docker/config/docker/docker-compose.yml |
| config/nginx/nginx.conf | config/nginx/nginx.conf |
| config/render/render.yaml | config/render/config/render/render.yaml |
| config/render/Procfile | config/render/config/render/Procfile |
| config/render/.env.render | config/render/config/render/.env.render |
| scripts/utils/debug_api.py | scripts/utils/scripts/utils/debug_api.py |
| scripts/deployment/quick_deploy.sh | scripts/deployment/scripts/deployment/quick_deploy.sh |
| scripts/utils/commit-changes.sh | scripts/utils/scripts/utils/commit-changes.sh |
| scripts/utils/generate_prospects_curl.py | scripts/utils/scripts/utils/generate_prospects_curl.py |
| api_logs.log | logs/api_logs.log |
| test_report.html | (removido) |
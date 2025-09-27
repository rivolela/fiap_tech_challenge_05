# Solução para Problemas com Logs da API

Foi identificado um problema em que o arquivo de logs da API (`logs/api_logs.log`) não estava sendo atualizado quando a API era executada localmente. Isso foi resolvido com as seguintes modificações:

## Alterações Implementadas

1. **Melhoria no Código de Inicialização de Logs**
   - Modificações no arquivo `src/api/scoring_api.py`
   - Priorização do caminho `logs/api_logs.log` para garantir consistência
   - Adição de permissões explícitas (777 para diretórios e 666 para arquivos)
   - Tratamento mais robusto de erros

2. **Script de Diagnóstico e Correção**
   - Criação do script `scripts/fix_logs.sh`
   - Verifica e corrige permissões dos diretórios e arquivos de log
   - Testa a escrita no arquivo de log
   - Fornece recomendações para inicialização da API

## Como Resolver o Problema

### Método 1: Usando o Script de Correção

Execute o script de correção de logs para diagnosticar e corrigir automaticamente os problemas:

```bash
./scripts/fix_logs.sh
```

### Método 2: Correção Manual

Se preferir fazer a correção manualmente:

1. Crie o diretório de logs (se não existir):
   ```bash
   mkdir -p logs
   ```

2. Defina permissões de escrita para todos:
   ```bash
   chmod -R 777 logs
   ```

3. Se o arquivo de log já existir, atualize suas permissões:
   ```bash
   chmod 666 logs/api_logs.log
   ```

### Método 3: Especifique o Arquivo de Log ao Iniciar a API

Defina explicitamente o caminho do arquivo de log ao iniciar a API:

```bash
LOG_FILE=logs/api_logs.log python -m src.api.scoring_api
```

## Verificação

Para verificar se os logs estão sendo escritos corretamente:

1. Inicie a API com um dos métodos acima
2. Faça algumas requisições à API
3. Verifique se o arquivo de logs está sendo atualizado:
   ```bash
   tail -f logs/api_logs.log
   ```

## Configuração no Dockerfile e Docker Compose

Nos ambientes Docker, a configuração já está correta, pois:
1. O Dockerfile cria os diretórios necessários
2. O volume em docker-compose.yml mapeia os logs corretamente
3. As permissões são definidas adequadamente no contêiner

O problema ocorre apenas em ambiente local, especialmente em sistemas operacionais onde o usuário que executa a aplicação pode ser diferente do usuário que criou os diretórios.
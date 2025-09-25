# Guia para Resolver Problemas de Timeout no Render

## Problema Identificado
O erro `WORKER TIMEOUT` no Render ocorre quando um worker (processo Uvicorn/Gunicorn) leva muito tempo para responder a uma solicitação ou consome muita memória. O código 134 geralmente indica um problema de memória ou timeout.

## Soluções Implementadas

### 1. Otimização do Gunicorn/Uvicorn

Atualizamos as configurações do Gunicorn nos seguintes arquivos:

- `render.yaml` - Comando de inicialização com timeout aumentado e menos workers:
  ```yaml
  startCommand: python scripts/check_memory.py && gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT --timeout 120 --graceful-timeout 60 --keep-alive 5 --log-level debug src.api.scoring_api:app
  ```

- `Procfile` - Mesmas configurações para consistência:
  ```
  web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT --timeout 120 --graceful-timeout 60 --keep-alive 5 --log-level debug src.api.scoring_api:app
  ```

- `Dockerfile` - Script de inicialização com configurações otimizadas:
  ```bash
  gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT --timeout 120 --graceful-timeout 60 --keep-alive 5 --max-requests 1000 --max-requests-jitter 50 --worker-tmp-dir /dev/shm --log-level debug src.api.scoring_api:app
  ```

### 2. Novos Scripts de Suporte

- `scripts/check_memory.py`: Monitoramento de uso de memória e criação de diretórios necessários
- `scripts/startup.sh`: Script para configurar limites de recursos para o ambiente de execução
- `.env.render`: Variáveis de ambiente específicas para o Render

### 3. Otimização do Pipeline de Treinamento

- Atualizamos o endpoint `/run-pipeline` para usar `BackgroundTasks` do FastAPI ao invés de threads
- Adicionamos timeout ao comando `subprocess.run` para evitar execuções infinitas
- Melhor tratamento de erros e logging

### 4. Configurações de Recursos

No arquivo `render.yaml`, adicionamos variáveis de ambiente para otimizar o uso de recursos:
```yaml
envVars:
  - key: PYTHON_UNBUFFERED
    value: "1"
  - key: WEB_CONCURRENCY
    value: "2"
  - key: PORT
    value: 8000
```

### 5. Uso de BackgroundTasks para o Pipeline de Treinamento

Modificamos o endpoint `run_pipeline` para:
- Usar o gerenciador de tarefas assíncronas do FastAPI (`BackgroundTasks`)
- Evitar uso de threads personalizadas que podem causar problemas de memória
- Adicionar timeout na execução do processo

## Como Testar as Alterações

1. Faça deploy das alterações no Render
2. Monitore os logs em tempo real para verificar problemas de memória
3. Teste os endpoints principais, especialmente os que podem consumir mais recursos

## Monitoramento Adicional

O script `check_memory.py` agora é executado na inicialização e fornece informações úteis sobre o ambiente de execução e uso de memória. Verifique os logs do Render para ver essas informações quando o serviço iniciar.
# Guia para Integração do Monitoramento de Modelos

Este guia descreve como configurar e usar o sistema de monitoramento do modelo de scoring.

## Estrutura de Diretórios

```
data/
  monitoring/             # Diretório principal de monitoramento
    model_metrics.json    # Métricas do modelo e histórico
    predictions_log.csv   # Log de predições realizadas
    drift_reports.json    # Relatórios de drift do modelo
    tmp/                  # Diretório para arquivos temporários
```

## Inicialização do Sistema de Monitoramento

Execute o script de inicialização para criar a estrutura necessária:

```bash
./scripts/initialize_metrics.sh
```

Este script cria:
1. Diretórios necessários
2. Arquivo inicial de métricas do modelo
3. Log de predições vazio
4. Arquivo inicial de relatórios de drift

## Visualização do Dashboard

Para iniciar o dashboard de monitoramento:

```bash
streamlit run src/dashboard/dashboard.py
```

## Endpoints de Monitoramento da API

A API fornece os seguintes endpoints para monitoramento:

- `GET /monitoring/metrics` - Obtém métricas atuais do modelo
- `GET /monitoring/drift` - Obtém análise de drift do modelo
- `GET /monitoring/predictions` - Obtém histórico de predições recentes
- `POST /monitoring/run-pipeline` - Inicia o pipeline de treinamento do modelo
- `GET /monitoring/pipeline-status` - Verifica o status do pipeline de treinamento

## Troubleshooting

### Se o dashboard não mostrar dados:

1. Verifique se os arquivos de métricas existem:
   ```bash
   ls -la data/monitoring/
   ```

2. Inicialize os arquivos de métricas se necessário:
   ```bash
   ./scripts/initialize_metrics.sh
   ```

3. Verifique permissões de arquivo:
   ```bash
   chmod -R 755 data/monitoring/
   ```

4. Verifique logs de erro:
   ```bash
   cat data/logs/dashboard.log
   ```

### Problemas de Timeout no Render

Se enfrentar problemas de "WORKER TIMEOUT" no ambiente Render:

1. A configuração no `render.yaml` foi otimizada:
   - Reduzido para 2 workers para economizar memória
   - Timeout aumentado para 120 segundos
   - Adicionada configuração de graceful-timeout (60s) e keep-alive (5s)

2. Limitações a serem consideradas:
   - O plano gratuito do Render tem limite de 512MB de memória
   - Cada worker do Gunicorn consome aproximadamente 200MB
   - Processos paralelos intensivos devem ser evitados

3. Recomendações:
   - Use `BackgroundTasks` do FastAPI para operações assíncronas
   - Limite o processamento em lote
   - Monitore o uso de memória com o script `scripts/check_memory.py`

## Adicionando Dados de Monitoramento

Novos dados de monitoramento são adicionados automaticamente quando:

1. Novas predições são realizadas através da API
2. O modelo é retreinado pelo pipeline
3. Análises de drift são executadas

Você também pode adicionar métricas manualmente usando o endpoint:
```bash
curl -X POST "http://localhost:8000/monitoring/metrics" -H "X-API-Key: YOUR_API_KEY" -H "Content-Type: application/json" -d '{"accuracy": 0.95, "precision": 0.82}'
```
# Implementação de Endpoints de Pipeline para o Dashboard

Este documento descreve como implementar corretamente os endpoints de pipeline que atualmente estão faltando na API, para que o dashboard de monitoramento possa funcionar adequadamente.

## Endpoints Faltantes

Atualmente, o dashboard tenta acessar os seguintes endpoints que não estão implementados na API:

1. `/monitoring/run-pipeline` - Para iniciar o pipeline de treinamento
2. `/monitoring/pipeline-status` - Para verificar o status do pipeline

## Solução Temporária

Como solução temporária, modificamos o código do dashboard para simular o comportamento desses endpoints sem fazer chamadas reais à API. Isso permite que os botões funcionem sem gerar erros para o usuário.

## Implementação Completa

Para implementar corretamente esses endpoints, siga as instruções abaixo:

### 1. Adicionar os Endpoints ao Router de Monitoramento

Edite o arquivo `src/api/monitoring_endpoints.py` e adicione os seguintes endpoints:

```python
@router.post(
    "/run-pipeline",
    status_code=status.HTTP_202_ACCEPTED,
    description="Executa o pipeline de treinamento do modelo"
)
async def run_pipeline(
    request: Request, 
    background_tasks: BackgroundTasks,
    role: str = Depends(verify_api_key)
):
    """
    Endpoint para executar o pipeline de treinamento do modelo.
    Requer autenticação com API key com nível 'admin'.
    """
    # Verificar se o usuário tem permissão de admin
    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permissão insuficiente para executar o pipeline. Requer API key com nível 'admin'."
        )
    
    try:
        # Variável global para manter o estado do pipeline
        if not hasattr(run_pipeline, "running"):
            run_pipeline.running = False
            run_pipeline.start_time = None
            run_pipeline.end_time = None
            run_pipeline.success = None
            run_pipeline.output = None
        
        # Verificar se o pipeline já está em execução
        if run_pipeline.running:
            return {
                "status": "in_progress",
                "message": "Pipeline de treinamento já está em execução.",
                "start_time": run_pipeline.start_time.isoformat() if run_pipeline.start_time else None
            }
        
        # Executar o script run_pipeline.sh em background usando BackgroundTasks
        def run_script():
            logger.info("Iniciando execução do pipeline de treinamento...")
            run_pipeline.running = True
            run_pipeline.start_time = datetime.datetime.now()
            
            try:
                # Executa com timeout maior para evitar problemas
                result = subprocess.run(
                    ["./scripts/run_pipeline.sh"],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minutos de timeout
                )
                run_pipeline.output = {
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                run_pipeline.success = (result.returncode == 0)
                
                if result.returncode == 0:
                    logger.info("Pipeline de treinamento concluído com sucesso.")
                else:
                    logger.error(f"Erro ao executar pipeline: {result.stderr}")
            except subprocess.TimeoutExpired:
                logger.error("Timeout ao executar pipeline de treinamento")
                run_pipeline.success = False
                run_pipeline.output = {"error": "Timeout ao executar pipeline"}
            except Exception as e:
                logger.exception(f"Erro na execução do pipeline: {str(e)}")
                run_pipeline.success = False
                run_pipeline.output = {"error": str(e)}
            
            run_pipeline.running = False
            run_pipeline.end_time = datetime.datetime.now()
        
        # Usar BackgroundTasks para não bloquear a API
        background_tasks.add_task(run_script)
        
        return {
            "status": "accepted",
            "message": "Pipeline de treinamento iniciado em background. Este processo pode levar alguns minutos.",
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.exception(f"Erro ao iniciar pipeline de treinamento: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao iniciar pipeline de treinamento: {str(e)}"
        )

@router.get(
    "/pipeline-status",
    status_code=status.HTTP_200_OK,
    description="Verifica o status do pipeline de treinamento"
)
async def get_pipeline_status(
    request: Request,
    role: str = Depends(verify_api_key)
):
    """
    Endpoint para verificar o status do pipeline de treinamento.
    """
    # Verificar se o pipeline foi inicializado
    if not hasattr(run_pipeline, "running"):
        return {
            "status": "not_started",
            "message": "Pipeline não foi iniciado ainda."
        }
    
    # Retornar o status atual
    status_response = {
        "running": run_pipeline.running,
        "start_time": run_pipeline.start_time.isoformat() if run_pipeline.start_time else None,
        "end_time": run_pipeline.end_time.isoformat() if run_pipeline.end_time else None,
    }
    
    if not run_pipeline.running and run_pipeline.end_time:
        status_response["success"] = run_pipeline.success
        
        if role == "admin" and run_pipeline.output:
            status_response["output_summary"] = {
                "stdout_lines": len(run_pipeline.output.get("stdout", "").split("\n")),
                "stderr_lines": len(run_pipeline.output.get("stderr", "").split("\n")),
            }
    
    return status_response
```

### 2. Verificar o Script de Pipeline

Certifique-se de que o script `scripts/run_pipeline.sh` existe e está funcionando corretamente:

```bash
#!/bin/bash

# Script para executar o pipeline de treinamento do modelo

echo "Iniciando pipeline de treinamento..."

# Verificar ambiente
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo "✅ Ambiente virtual ativado."
fi

# Definir variáveis
DATA_DIR="data/processed"
OUTPUT_DIR="models"
LOG_FILE="logs/pipeline.log"

# Criar diretório de logs se não existir
mkdir -p logs

# Executar o pipeline
echo "$(date): Iniciando pipeline de treinamento" >> "$LOG_FILE"

echo "1. Preparando dados..."
python src/features/feature_engineering.py >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
  echo "❌ Erro na preparação dos dados. Verifique o log para mais detalhes."
  exit 1
fi

echo "2. Treinando modelo..."
python src/models/train_simple.py >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
  echo "❌ Erro no treinamento do modelo. Verifique o log para mais detalhes."
  exit 1
fi

echo "3. Gerando métricas e visualizações..."
python src/features/feature_importance.py >> "$LOG_FILE" 2>&1

echo "✅ Pipeline concluído com sucesso!"
echo "$(date): Pipeline concluído com sucesso" >> "$LOG_FILE"
```

### 3. Reverter as Alterações no Dashboard

Depois de implementar os endpoints, reverter as alterações temporárias no dashboard, removendo as simulações e descomentando o código que faz as chamadas reais para a API:

1. Edite `src/dashboard/dashboard.py` e restaure o código original dos botões "Treinar Modelo".
2. Restaure a função `check_pipeline_status()` para fazer chamadas reais à API.

### 4. Testando a Implementação

1. Reconstrua os contêineres:
   ```bash
   docker compose build --no-cache && docker compose down && docker compose up -d
   ```

2. Acesse o dashboard em http://localhost:8502
3. Clique no botão "Treinar Modelo" e verifique se o pipeline é iniciado corretamente
4. Verifique se o status do pipeline é atualizado corretamente

## Considerações

- Os endpoints implementados utilizam `BackgroundTasks` do FastAPI para executar o pipeline em segundo plano sem bloquear a API
- O status do pipeline é mantido em memória usando variáveis estáticas na função `run_pipeline`
- Para uma implementação mais robusta, considere usar um banco de dados para armazenar o status do pipeline, especialmente se houver múltiplas instâncias da API
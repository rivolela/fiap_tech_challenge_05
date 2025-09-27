# Scripts de Utilidade para o Projeto

Este diretório contém scripts organizados para facilitar o desenvolvimento, teste e implantação do projeto.

## Estrutura de Diretórios

### 📂 core/
Scripts principais para executar o projeto:
- `setup.sh` - Configura o ambiente de desenvolvimento
- `run_pipeline.sh` - Executa o pipeline completo de treinamento com MLflow
- `start_api.sh` - Inicia a API do modelo de scoring
- `retrain_model.sh` - Retreina o modelo com as versões atuais das bibliotecas
- `start_dashboard.sh` - Inicia o dashboard de monitoramento

### 📂 monitoring/
Scripts para monitoramento do modelo:
- `check_drift.sh` - Verifica drift do modelo e envia alertas
- `update_monitoring_metrics.py` - Atualiza métricas de monitoramento

### 📂 utils/
Scripts de utilidade:
- `generate_api_key.sh` - Gera chaves de API para o sistema
- `fix_paths.py` - Corrige problemas de caminhos no projeto

### 📂 tests/
Scripts para testes:
- `run_tests.sh` - Executa testes unitários com cobertura
- `test_llm_comments.py` - Testa comentários LLM

## Como Usar

Para iniciar o projeto pela primeira vez:
```bash
./scripts/core/setup.sh
```

Para executar o pipeline completo:
```bash
./scripts/core/run_pipeline.sh
```

Para iniciar a API:
```bash
./scripts/core/start_api.sh
```

Para executar testes:
```bash
./scripts/tests/run_tests.sh
```
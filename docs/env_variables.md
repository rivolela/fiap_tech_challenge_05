## Variáveis de Ambiente da API

A API de scoring pode ser configurada através das seguintes variáveis de ambiente:

| Variável | Descrição | Valor Padrão |
|----------|-----------|--------------|
| PORT | Porta onde a API será executada | 8000 |
| LOG_LEVEL | Nível de logging (DEBUG, INFO, WARNING, ERROR) | INFO |
| LOG_FILE | Caminho para o arquivo de log | logs/api_logs.log |
| API_KEYS | Lista de chaves de API no formato JSON | [] |
| API_KEY_SALT | Salt para geração de chaves de API | "" |
| RATE_LIMIT_WINDOW | Janela de tempo (segundos) para limite de requisições | 60 |
| RATE_LIMIT_MAX | Número máximo de requisições na janela de tempo | 30 |
| **CLASSIFICATION_THRESHOLD** | **Threshold para classificação de candidatos** | **0.25** |

### Threshold de Classificação

O threshold de classificação foi ajustado para melhorar o equilíbrio entre precisão e recall em conjuntos de dados desbalanceados. Um valor mais baixo (0.25 em vez do padrão 0.5) aumenta a chance de identificar candidatos qualificados.

Para ajustar o threshold:

```bash
# Localmente
export CLASSIFICATION_THRESHOLD=0.2
python -m src.api.scoring_api

# No Docker Compose
# Editar o valor no docker-compose.yml
```

O notebook `notebooks/threshold_adjustment_analysis.ipynb` contém uma análise detalhada do impacto de diferentes thresholds nas métricas de desempenho do modelo.
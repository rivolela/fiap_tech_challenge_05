# Guia de Monitoramento do Modelo Decision Scoring

Este documento explica como utilizar o sistema de monitoramento para acompanhar o desempenho e o drift do modelo de scoring da Decision.

## Sumário

1. [Visão Geral](#visão-geral)
2. [Configuração do Sistema de Monitoramento](#configuração-do-sistema-de-monitoramento)
3. [Dashboard de Métricas](#dashboard-de-métricas)
4. [API de Monitoramento](#api-de-monitoramento)
5. [Alertas e Notificações](#alertas-e-notificações)
6. [Detecção de Drift do Modelo](#detecção-de-drift-do-modelo)
7. [Manutenção e Solução de Problemas](#manutenção-e-solução-de-problemas)

## Visão Geral

O sistema de monitoramento do modelo Decision Scoring foi projetado para acompanhar de forma contínua o desempenho do modelo em produção, detectar desvios (drift) nos dados e fornecer alertas sobre problemas potenciais. O sistema consiste em:

- **Coleta de métricas**: Armazenamento automático de métricas de desempenho do modelo.
- **Detecção de drift**: Comparação contínua entre distribuições de dados de treino e produção.
- **Dashboard visual**: Interface para visualização das métricas e tendências.
- **Sistema de alertas**: Notificação quando métricas críticas caem abaixo de limites aceitáveis.

## Configuração do Sistema de Monitoramento

### Requisitos

- Python 3.8 ou superior
- Dependências: streamlit, pandas, matplotlib, seaborn, fastapi

### Instalação

1. As dependências são instaladas automaticamente quando você executa `./scripts/setup.sh`.

2. Caso necessite instalar manualmente:
```bash
pip install streamlit pandas matplotlib seaborn
```

### Inicialização do Dashboard

Para iniciar o dashboard de monitoramento:

```bash
./scripts/start_dashboard.sh
```

O dashboard será iniciado em http://localhost:8501 por padrão.

## Dashboard de Métricas

O dashboard está dividido em quatro seções principais:

### Dashboard Principal

Visão geral das métricas principais do modelo e status de drift.

![Dashboard Principal](data/monitoring/docs/dashboard_main.png)

### Análise de Drift

Detalhes sobre o drift do modelo, incluindo:
- Score de drift geral
- Features com drift detectado
- Visualizações comparativas entre distribuições de treino e produção

![Análise de Drift](data/monitoring/docs/dashboard_drift.png)

### Histórico de Métricas

Evolução das métricas do modelo ao longo do tempo:
- Acurácia
- Precisão
- Recall
- F1-Score

![Histórico de Métricas](data/monitoring/docs/dashboard_metrics.png)

### Predições Recentes

Análise das predições realizadas pelo modelo:
- Taxa de aprovação
- Distribuição por segmento
- Evolução temporal das predições

![Predições Recentes](data/monitoring/docs/dashboard_predictions.png)

## API de Monitoramento

O sistema expõe endpoints de API para consulta programática das métricas e drift do modelo.

### Endpoints Disponíveis

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/metrics/` | GET | Métricas básicas da API e modelo |
| `/monitoring/drift` | GET | Análise completa de drift do modelo |
| `/monitoring/drift/visualization` | GET | Visualização de drift para uma feature específica |
| `/monitoring/metrics/history` | GET | Histórico de métricas do modelo |
| `/monitoring/predictions/recent` | GET | Estatísticas sobre predições recentes |

### Exemplos de Uso

Consultar métricas do modelo:
```bash
curl -H "X-API-Key: sua-api-key" http://localhost:8000/metrics/
```

Verificar drift do modelo:
```bash
curl -H "X-API-Key: sua-api-key" http://localhost:8000/monitoring/drift
```

## Alertas e Notificações

O sistema de alertas monitora continuamente as métricas e o drift do modelo, notificando quando valores críticos são atingidos.

### Configuração de Alertas

Os alertas são configurados no arquivo `data/monitoring/alerts/alert_config.json`:

```json
{
  "alert_recipients": ["email@exemplo.com"],
  "thresholds": {
    "drift_score": 0.3,
    "accuracy_drop": 0.05,
    "error_rate": 1.0,
    "latency": 200
  },
  "smtp_config": {
    "enabled": false,
    "server": "smtp.exemplo.com",
    "port": 587,
    "user": "username",
    "password": "senha",
    "from_email": "alerts@decision.com"
  }
}
```

### Tipos de Alertas

1. **Alertas de Drift**: Notificam quando o score de drift ultrapassa o limiar definido.
2. **Alertas de Desempenho**: Notificam quando a acurácia ou outras métricas caem abaixo de limiares aceitáveis.
3. **Alertas de Operação**: Notificam sobre problemas operacionais da API (latência elevada, taxa de erro).

## Detecção de Drift do Modelo

### Metodologia

O sistema utiliza as seguintes técnicas para detectar drift:

1. **Drift em Features Numéricas**: Detectado usando a diferença padronizada entre médias (distância entre média atual e média de treino em unidades de desvio padrão).

2. **Drift em Features Categóricas**: Detectado usando divergência Jensen-Shannon entre distribuições de categorias.

3. **Score de Drift Geral**: Calculado como a proporção de features com drift detectado.

### Interpretação de Resultados

- **Score de Drift < 0.1**: O modelo está estável.
- **Score de Drift entre 0.1 e 0.3**: Monitorar de perto.
- **Score de Drift > 0.3**: Drift significativo detectado, considerar retreinamento do modelo.

## Manutenção e Solução de Problemas

### Manutenção de Rotina

1. **Backup de Dados de Monitoramento**:
   ```bash
   # Backup de dados de monitoramento
   cp -r data/monitoring /backup/monitoring_$(date +"%Y%m%d")
   ```

2. **Limpeza de Dados Antigos**:
   O sistema mantém dados por um período limitado. Para limpeza manual:
   ```bash
   # Limpar dados de predições mais antigos que 90 dias
   python -c "from src.monitoring.metrics_store import clean_old_predictions; clean_old_predictions(days=90)"
   ```

### Solução de Problemas Comuns

1. **Dashboard não inicia**:
   - Verificar se o Streamlit está instalado: `pip show streamlit`
   - Verificar se os diretórios de dados existem: `ls -la data/monitoring/`

2. **Alertas não estão sendo enviados**:
   - Verificar configuração SMTP em `data/monitoring/alerts/alert_config.json`
   - Testar configuração de email: `python -c "from src.monitoring.alert_system import test_email_connection; test_email_connection()"`

3. **Análise de drift retorna erro**:
   - Verificar se há estatísticas de treinamento: `ls -la data/monitoring/training_statistics.json`
   - Verificar se há predições registradas: `ls -la data/monitoring/predictions_log.csv`

## Contatos e Suporte

Para questões ou suporte relacionado ao sistema de monitoramento:
- Email: suporte@decision.com
- Documentação completa: `/docs/monitoring_guide.md`
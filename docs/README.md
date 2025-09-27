# Documentação do Projeto Decision Scoring

Este diretório contém a documentação do projeto Decision Scoring, organizados por tópicos.

## Índice de Documentos

| Documento | Descrição |
|-----------|-----------|
| [API_README.md](./API_README.md) | Documentação principal da API de Scoring, incluindo endpoints e como usá-los |
| [monitoring_guide.md](./monitoring_guide.md) | Guia completo para o sistema de monitoramento do modelo |
| [mlflow_guide.md](./mlflow_guide.md) | Guia para uso do MLflow no projeto para experimentação e rastreamento de modelos |
| [llm_comments.md](./llm_comments.md) | Documentação sobre os comentários gerados por LLM nas predições |
| [resumo_solucoes.md](./resumo_solucoes.md) | Resumo de soluções implementadas para problemas encontrados no projeto |

## Estrutura do Projeto

O projeto Decision Scoring está organizado da seguinte forma:

```
fiap_tech_challenge_05/
├── data/               # Dados brutos, processados e de monitoramento
├── docs/               # Documentação (este diretório)
├── logs/               # Arquivos de log gerados pela aplicação
├── models/             # Modelos treinados serializados
├── notebooks/          # Jupyter notebooks para análises exploratórias
├── scripts/            # Scripts utilitários organizados por finalidade
│   ├── core/           # Scripts principais (setup, pipeline, API)
│   ├── monitoring/     # Scripts de monitoramento
│   ├── tests/          # Scripts de teste
│   └── utils/          # Scripts utilitários
├── src/                # Código-fonte da aplicação
│   ├── api/            # Código da API (FastAPI)
│   ├── data/           # Módulos para manipulação de dados
│   ├── features/       # Engenharia de features
│   ├── models/         # Treinamento e avaliação de modelos
│   └── monitoring/     # Código para monitoramento de modelos
└── tests/              # Testes unitários e de integração
```

## Como Contribuir

Para contribuir com a documentação:

1. Atualize os documentos existentes quando implementar mudanças relacionadas
2. Mantenha o estilo consistente usando Markdown
3. Atualize este índice quando adicionar novos documentos
4. Use exemplos claros e instruções passo a passo onde aplicável
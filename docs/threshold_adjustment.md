# Ajuste de Threshold de Classificação

## Problema

O modelo de scoring está apresentando desequilíbrio entre precisão e recall:
- **Precisão muito alta**: ~98%
- **Recall muito baixo**: ~14-16% 

Este comportamento é típico de modelos treinados com dados desbalanceados, onde a classe positiva (candidatos recomendados) é muito menos frequente que a classe negativa.

## Solução Implementada

Foi implementado um sistema de threshold ajustável para a classificação:

1. **Threshold Reduzido**: O valor padrão foi ajustado de 0.5 para 0.25
2. **Configuração Flexível**: O threshold pode ser configurado através da variável de ambiente `CLASSIFICATION_THRESHOLD`
3. **Transparência nos Resultados**: A API agora retorna o threshold utilizado junto com a predição

### Impacto do Ajuste

Um threshold mais baixo:
- **Aumenta o recall**: Mais candidatos qualificados serão recomendados
- **Pode reduzir ligeiramente a precisão**: Mas dada a alta precisão inicial (98%), há margem para esse trade-off

## Como Configurar o Threshold

### 1. Localmente via Variável de Ambiente

```bash
# Definir o threshold antes de iniciar a API
export CLASSIFICATION_THRESHOLD=0.2
uvicorn src.api.scoring_api:app --reload
```

### 2. Em Ambientes Docker via docker-compose.yml

```yaml
services:
  scoring-api:
    image: scoring-api:latest
    environment:
      - CLASSIFICATION_THRESHOLD=0.25
```

### 3. Em Ambientes de Produção (Render)

Adicione a variável de ambiente `CLASSIFICATION_THRESHOLD` nas configurações do serviço no Render Dashboard.

## Como Testar Diferentes Thresholds

Foi criado um script para testar diferentes thresholds e avaliar o impacto:

```bash
./scripts/test_thresholds.sh
```

O script testa thresholds de 0.5 (padrão original), 0.3, 0.2 e 0.1, permitindo comparar como a classificação muda para diferentes valores.

## Recomendações de Valor

- **Threshold = 0.5**: Comportamento tradicional (status quo)
- **Threshold = 0.3**: Equilíbrio moderado (recomendado inicialmente)
- **Threshold = 0.2**: Favorece recall, bom para primeira triagem
- **Threshold = 0.1**: Muito permissivo, pode gerar muitos falsos positivos

A escolha ideal do threshold deve ser baseada nas necessidades específicas do negócio:
- Se o custo de perder um bom candidato é alto → use threshold mais baixo
- Se o custo de entrevistar candidatos inadequados é alto → use threshold mais alto
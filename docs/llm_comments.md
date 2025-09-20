# API de Scoring com Comentários LLM

Esta API fornece um sistema de scoring para candidatos a vagas, incluindo comentários personalizados gerados com técnicas de LLM (Large Language Models).

## Funcionalidades Recentes

### Comentários Personalizados via LLM

A API agora inclui uma funcionalidade de comentários personalizados para cada recomendação de candidato. Estes comentários são gerados usando técnicas simples de Processamento de Linguagem Natural (NLP) via TextBlob.

#### Benefícios:

- **Explicabilidade:** Ajuda recrutadores a entenderem o motivo da recomendação
- **Personalização:** Comentários adaptados ao perfil do candidato e vaga
- **Linguagem Natural:** Formatação em texto claro e compreensível 

#### Características:

- Análise de compatibilidade entre candidato e vaga
- Consideração de fatores como experiência, educação e área de formação
- Variação nos templates para evitar repetição
- Diferentes formatos para recomendações positivas e negativas

#### Exemplo de Resposta API:

```json
{
  "prediction": 1,
  "probability": 0.85,
  "recommendation": "Recomendado",
  "comment": "A avaliação técnica sugere boa adequação para a função de Desenvolvedor Python na área de tecnologia, nível pleno. Destaca-se formação superior na área de tecnologia e 5.0 anos de experiência relevante.",
  "vaga_info": {
    "id": "vaga-123",
    "titulo": "Desenvolvedor Python",
    "area": "tecnologia",
    "senioridade": "pleno"
  },
  "match_score": 0.78
}
```

## Como Usar

Para receber comentários personalizados, forneça informações detalhadas sobre a vaga na requisição:

```json
{
  "idade": 28,
  "experiencia": 5,
  "educacao": "ensino_superior",
  "area_formacao": "tecnologia",
  "vaga_titulo": "Desenvolvedor Python",
  "vaga_area": "tecnologia",
  "vaga_senioridade": "pleno"
}
```

## Implementação Técnica

- Utiliza a biblioteca **TextBlob** para processamento de linguagem natural
- Templates variados para diferentes situações de recomendação
- Lógica para selecionar comentários baseados no perfil e na probabilidade
- Fallback para comentários genéricos em caso de erro ou dados insuficientes

## Evolução Futura

Em versões futuras, planejamos:

- Integração com LLMs mais sofisticados
- Geração de comentários mais detalhados e específicos
- Sugestões de próximos passos para recrutadores
- Análise mais profunda de compatibilidade de habilidades
# Implantação da Decision Scoring API no Render

Este guia explica como implantar a API de scoring Decision na plataforma Render.

## Pré-requisitos

1. Uma conta no [Render](https://render.com/)
2. Seu código em um repositório Git (GitHub, GitLab, etc.)

## Métodos de Implantação

### Opção 1: Implantação Automática via Blueprint (recomendado)

1. Faça login no Render
2. Clique em "New +" e selecione "Blueprint"
3. Conecte seu repositório GitHub/GitLab
4. Selecione o repositório `fiap_tech_challenge_05`
5. O Render detectará automaticamente o arquivo `render.yaml` e configurará seu serviço

### Opção 2: Implantação Manual via Docker

1. Faça login no Render
2. Clique em "New +" e selecione "Web Service"
3. Conecte seu repositório GitHub/GitLab
4. Configure o serviço:
   - **Nome**: decision-scoring-api
   - **Ambiente**: Docker
   - **Caminho do Dockerfile**: ./Dockerfile
   - **Branch**: main
   - **Comando**: gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT src.api.scoring_api:app
   - **Health Check Path**: /health
   - **Plano**: Free (ou outro conforme necessidade)
5. Clique em "Create Web Service"

### Opção 3: Implantação sem Docker

1. Faça login no Render
2. Clique em "New +" e selecione "Web Service"
3. Conecte seu repositório GitHub/GitLab
4. Configure o serviço:
   - **Nome**: decision-scoring-api
   - **Ambiente**: Python 3
   - **Build Command**: pip install -r requirements.txt
   - **Start Command**: gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT src.api.scoring_api:app
   - **Health Check Path**: /health
   - **Plano**: Free (ou outro conforme necessidade)
5. Clique em "Create Web Service"

## Variáveis de Ambiente

Configure as seguintes variáveis de ambiente:

- `PORT`: 8000 (ou deixe o Render configurar automaticamente)
- `PYTHONPATH`: /app (para Docker) ou seu diretório raiz (sem Docker)

## Verificação de Implantação

Após a implantação, verifique se a API está funcionando corretamente:

1. Acesse `https://sua-api-url.onrender.com/health` para verificar a saúde da API
2. Acesse `https://sua-api-url.onrender.com/docs` para verificar a documentação da API

## Notas Importantes

- O plano gratuito do Render pode "adormecer" após períodos de inatividade
- O primeiro acesso pode ser lento enquanto o serviço é iniciado
- Recomenda-se criar uma variável de ambiente para as API Keys em produção, em vez de hardcoded
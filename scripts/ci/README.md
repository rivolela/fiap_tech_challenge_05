# Scripts de CI/CD para o Projeto Decision Scoring

Este diretório contém scripts para automação de processos de Integração Contínua (CI) e Entrega Contínua (CD).

## Scripts Disponíveis

### 1. test_commit_deploy.sh

Script para desenvolvedores executarem o processo de teste, commit e deploy em um único comando.

**Uso:**
```bash
./test_commit_deploy.sh ["Mensagem do commit"]
```

Se não for fornecida uma mensagem de commit, o script tentará gerar automaticamente uma mensagem usando um modelo LLM (GPT) com base nas alterações detectadas no repositório.

**Requisitos para geração automática de mensagens:**
- Arquivo `.env` na raiz do projeto com a variável `OPENAI_API_KEY` definida
- Conexão com a internet para acessar a API do OpenAI
- Alterações detectáveis no repositório git

**Exemplo de mensagem gerada:**
```
feat(api): adiciona validação de entrada nos endpoints de scoring
```

### 2. generate_commit_message.sh

Script específico para gerar automaticamente mensagens de commit baseadas nas alterações, usando um modelo LLM (GPT).

**Uso:**
```bash
./generate_commit_message.sh
```

**Funcionalidades:**
- Analisa as alterações no repositório git (staged e unstaged)
- Gera uma mensagem de commit no formato Conventional Commits
- Permite aceitar a mensagem e fazer commit diretamente
- Se rejeitada, copia a mensagem para a área de transferência

**Requisitos:**
- Arquivo `.env` na raiz do projeto com a variável `OPENAI_API_KEY` definida
- Conexão com a internet para acessar a API do OpenAI

### 3. deploy.sh

**Funcionalidades:**
- Executa os testes do projeto
- Faz commit das alterações com a mensagem fornecida
- Realiza push para o branch atual
- Informa sobre o status do deploy (se configurado)

### 2. deploy.sh

Script simplificado para deploy em ambientes como Render, onde pode ser configurado como comando de inicialização.

**Uso:**
```bash
APP_ENV=production ./deploy.sh
```

**Variáveis de ambiente:**
- `APP_ENV`: Define o ambiente (development, staging, production)

**Funcionalidades:**
- Executa testes (quando disponíveis)
- Aplica migrações de banco de dados (se configurado)
- Inicia a aplicação no modo apropriado para o ambiente

## Configuração do CI/CD no GitHub Actions

Este projeto também inclui um workflow para GitHub Actions em `.github/workflows/ci-cd.yml` que automatiza:

1. Execução de testes em cada push e pull request
2. Deploy para ambiente de staging quando há push para a branch develop
3. Deploy para ambiente de produção quando há push para a branch main

### Configuração necessária:

Para que o CI/CD funcione corretamente, você precisa configurar os seguintes secrets no seu repositório GitHub:

- `RENDER_API_KEY`: Chave de API do Render
- `RENDER_SERVICE_ID`: ID do serviço de staging no Render
- `RENDER_PROD_SERVICE_ID`: ID do serviço de produção no Render

## Fluxo de Trabalho Recomendado

1. Desenvolva em uma branch de feature
2. Execute `./scripts/ci/test_commit_deploy.sh "Sua mensagem"` para testar e commitar
3. Crie um Pull Request para a branch develop
4. Após revisão e aprovação, faça merge para develop (deploy em staging)
5. Valide em staging e faça merge de develop para main (deploy em produção)
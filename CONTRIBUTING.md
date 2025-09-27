# Contribuindo para o Projeto Decision Scoring

Agradecemos seu interesse em contribuir para o projeto! Aqui estão algumas diretrizes para ajudar você a começar.

## Fluxo de Trabalho

1. Faça fork do repositório
2. Clone o fork para sua máquina local
3. Configure o repositório upstream
4. Crie um branch para suas alterações
5. Faça suas alterações
6. Envie suas alterações para seu fork
7. Crie um Pull Request

## Convenções de Código

- Use snake_case para nomes de variáveis e funções
- Use PascalCase para nomes de classes
- Use UPPER_CASE para constantes
- Siga o padrão PEP 8 para código Python

## Estrutura do Projeto

Respeite a estrutura de diretórios do projeto:

```
fiap_tech_challenge_05/
├── .github/              # GitHub Actions e templates para issues/PRs
├── config/               # Arquivos de configuração
│   ├── docker/           # Arquivos Docker
│   ├── nginx/            # Configuração Nginx
│   └── render/           # Configuração da plataforma Render
├── data/                 # Dados do projeto
├── docs/                 # Documentação
├── examples/             # Exemplos de uso
├── logs/                 # Logs da aplicação
├── models/               # Modelos treinados
├── notebooks/            # Jupyter notebooks
├── scripts/              # Scripts utilitários
│   ├── deployment/       # Scripts de implantação
│   ├── monitoring/       # Scripts de monitoramento
│   └── utils/            # Scripts utilitários diversos
├── src/                  # Código-fonte principal
└── tests/                # Testes
```

## Commits

Use mensagens de commit claras e descritivas, seguindo o padrão:

```
tipo: descrição concisa

Descrição mais detalhada se necessário.
```

Onde `tipo` pode ser:
- `feat`: Nova funcionalidade
- `fix`: Correção de bug
- `docs`: Alterações na documentação
- `style`: Formatação, ponto e vírgula, etc; sem alteração de código
- `refactor`: Refatoração de código
- `test`: Adicionando testes, refatorando testes
- `chore`: Atualizações de tarefas de build, configurações, etc

## Testes

Certifique-se de adicionar testes para qualquer funcionalidade nova ou corrigida.
Execute a suite de testes antes de enviar seu Pull Request.

```bash
./scripts/run_tests.sh
```

## Documentação

Atualize a documentação relevante para suas alterações:

- README.md para alterações de uso
- Docstrings para funções e classes
- Comentários para código complexo

## Revisão de Código

- Todo código deve passar por revisão antes de ser mesclado
- Responda prontamente a comentários e solicitações de alteração
- Mantenha discussões técnicas construtivas

## Padrões de Codificação

### Python

- Utilize tipagem estática quando possível (ex: `def function(param: str) -> bool:`)
- Escreva docstrings no formato Google ou NumPy para todas as funções e classes
- Mantenha funções pequenas e focadas em uma única responsabilidade
- Use nomes descritivos para variáveis e funções

### Testes

- Nomeie testes descritivamente: `test_should_return_400_when_missing_required_field`
- Organize testes em classes por funcionalidade
- Use fixtures e mocks para facilitar testes

### Registro de Logs

- Use níveis apropriados de log: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Inclua contexto suficiente em mensagens de log
- Configure logs para ir para o diretório `logs/`
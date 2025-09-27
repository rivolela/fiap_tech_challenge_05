## 🧪 Testes Unitários

O projeto conta com uma suíte de testes unitários para garantir a qualidade e o funcionamento adequado de cada componente da pipeline.

### Estrutura de Testes

A estrutura de testes segue a organização do código-fonte:

```
tests/
├── unit/
│   ├── api/
│   │   ├── test_model_loader.py
│   │   ├── test_preprocessing.py
│   │   ├── test_security.py
│   │   └── ...
│   ├── features/
│   │   ├── test_feature_engineering.py
│   │   ├── test_data_validation.py
│   │   ├── test_cross_validation.py
│   │   └── ...
│   └── models/
│       ├── test_train_simple.py
│       ├── test_mlflow_server.py
│       └── ...
└── integration/
    └── ...
```

### Executando os Testes

Para executar os testes unitários e gerar relatório de cobertura:

```bash
# Execute o script de testes
./scripts/run_tests.sh
```

Ou manualmente:

```bash
# Instale as dependências de desenvolvimento
pip install -r requirements-dev.txt

# Execute os testes com relatório de cobertura
python -m pytest tests/unit -v --cov=src --cov-report=term --cov-report=html
```

O relatório de cobertura HTML será gerado no diretório `htmlcov/`. Abra o arquivo `htmlcov/index.html` em um navegador para visualizar os resultados detalhados.

### Dependências de Testes

As dependências necessárias para executar os testes estão listadas no arquivo `requirements-dev.txt`:

- pytest: Framework de testes
- pytest-cov: Plugin para cobertura de código
- pytest-mock: Plugin para mock de objetos
- coverage: Ferramenta para análise de cobertura
- flake8: Ferramenta de linting
- black: Formatador de código
- isort: Organizador de imports
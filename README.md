# FIAP Tech Challenge 05 - Análise de Dados de Recrutamento

Este projeto contém análises exploratórias de dados para sistema de recrutamento utilizando Python e Jupyter Notebooks.

## 🚀 Configuração do Ambiente

### Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Configuração Automática
Execute o script de setup:
```bash
./setup.sh
```

### Configuração Manual
1. Criar ambiente virtual:
```bash
python3 -m venv .venv
```

2. Ativar ambiente virtual:
```bash
source .venv/bin/activate  # macOS/Linux
# ou
.venv\Scripts\activate     # Windows
```

3. Instalar dependências:
```bash
pip install -r requirements.txt
```

4. Verificar instalação:
```bash
python check_env.py
```

## 📊 Estrutura do Projeto

```
fiap_tech_challenge_05/
├── notebooks/
│   └── Análise Exploratória dos Dados.ipynb
│   └── analise_exploratoria.ipynb
├── data/
│   ├── raw/
│   │   ├── applicants.json
│   │   ├── prospects.json
│   │   └── jobs.json
│   ├── processed/
│   │   ├── complete_processed_data.csv
│   │   └── splits/
│   │       ├── X_train.csv
│   │       ├── X_test.csv
│   │       └── ...
│   ├── insights/
│   └── visualizations/
├── src/
│   ├── data/
│   │   ├── data_analysis.py
│   │   └── download_data.py
│   ├── features/
│   │   ├── data_validation.py
│   │   └── feature_engineering.py
│   └── models/
│       ├── train_simple.py
│       ├── train_model.py
│       └── mlflow_server.py
├── models/
│   ├── scoring_model.pkl
│   └── feature_scaler.pkl
├── docs/
│   └── mlflow_guide.md
├── scripts/
│   └── check_env.py
├── requirements.txt
├── setup.sh
├── check_env.py
├── .gitignore
├── LICENSE
└── README.md
```

## 🔧 Bibliotecas Utilizadas

- **pandas**: Manipulação e análise de dados
- **numpy**: Computação científica
- **matplotlib**: Visualização de dados
- **seaborn**: Visualização estatística
- **plotly**: Gráficos interativos
- **scikit-learn**: Machine learning
- **jupyter**: Ambiente de notebooks
- **mlflow**: Rastreamento e gerenciamento de experimentos de ML

## 📈 Como Usar

1. Ativar o ambiente virtual:
```bash
source .venv/bin/activate
```

2. Iniciar Jupyter Notebook:
```bash
jupyter notebook
```

3. Abrir o notebook [`notebooks/Análise Exploratória dos Dados.ipynb`](notebooks/Análise%20Exploratória%20dos%20Dados.ipynb)

## 🤖 Sistema Híbrido de Scoring + Clustering

### Pipeline Completo

Execute o pipeline completo com MLflow usando:

```bash
./run_pipeline.sh
```

Opções disponíveis:
- `--compare`: Treina e compara diferentes modelos
- `--port 8080`: Altera a porta do servidor MLflow (padrão: 5001)
- `--no-server`: Executa o pipeline sem iniciar o servidor MLflow

### Treinamento do Modelo

Para treinar modelos separadamente:

```bash
# Treinar o modelo RandomForest padrão
python src/models/train_simple.py

# Comparar diferentes modelos
python src/models/train_simple.py --compare
```

### MLflow - Tracking de Experimentos

Para gerenciar experimentos MLflow:

```bash
# Iniciar servidor MLflow
python src/models/mlflow_server.py

# Listar experimentos
python src/models/mlflow_server.py --list

# Excluir experimento
python src/models/mlflow_server.py --delete "Decision-Scoring-Model"
```

Para mais detalhes sobre MLflow, consulte o guia em [docs/mlflow_guide.md](docs/mlflow_guide.md)

## 📝 Dados

O projeto utiliza três datasets em formato JSON na pasta `data/`:
- **applicants.json**: Dados dos candidatos ao processo seletivo
- **prospects.json**: Dados das entrevistas e prospects
- **vagas.json**: Informações sobre as vagas disponíveis

### Estrutura dos Dados
- **Candidatos**: Dataset principal com informações dos candidatos
- **Entrevistas**: Dados das entrevistas realizadas  
- **Vagas**: Informações sobre as vagas disponíveis

## 🔍 Análise Exploratória

O notebook principal inclui:
- Importação e carregamento dos dados JSON
- Análise da estrutura e qualidade dos dados
- Estatísticas descritivas
- Visualizações gráficas interativas
- Identificação de padrões e outliers
- Análise de correlações
- Detecção de valores ausentes

### Principais Insights
- Análise de padrões nos dados de candidatos
- Identificação de correlações entre variáveis
- Detecção e tratamento de outliers
- Distribuições das variáveis principais

## 🔮 Próximos Passos

1. **Experimentação com MLflow**: 
   - Otimização de hiperparâmetros
   - Teste de diferentes algoritmos
   - Comparação de métricas de desempenho

2. **Implementação de Clustering**:
   - Segmentação de candidatos por perfil
   - Identificação de grupos de vagas similares
   - Integração do clustering ao scoring model

3. **Melhorias no Modelo**:
   - Feature engineering avançado
   - Implementação de técnicas de deep learning
   - Validação cruzada para maior robustez

4. **Produtivização**:
   - API REST para servir o modelo
   - Monitoramento contínuo de performance
   - Pipeline de retreinamento automático

## 🤝 Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença Apache 2.0. Veja o arquivo [`LICENSE`](LICENSE) para mais detalhes.

## 📞 Contato

FIAP Tech Challenge 05 - [@rivolela](https://github.com/rivolela/fiap_tech_challenge_05)

---
**Nota**: Certifique-se de que os arquivos JSON estejam disponíveis na pasta `data/` antes de executar o notebook de análise.

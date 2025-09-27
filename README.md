# FIAP Tech Challenge 05 - Decision Scoring API

Este projeto contém análises exploratórias de dados para sistema de recrutamento e uma API de scoring para auxiliar na tomada de decisão no processo de recrutamento.

## 🚀 Configuração do Ambiente

### Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Configuração Automática
Execute o script de setup:
```bash
./scripts/setup.sh
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
│   └── threshold_adjustment_analysis.ipynb
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
- **fastapi**: Framework para API
- **jupyter**: Ambiente de notebooks
- **mlflow**: Rastreamento e gerenciamento de experimentos de ML
- **textblob**: Processamento de linguagem natural para LLM

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

## 🛠️ Variáveis de Ambiente

Para configurar a API, você pode usar diversas variáveis de ambiente:
- `PORT`: Porta onde a API será executada (padrão: 8000)
- `LOG_LEVEL`: Nível de logging (padrão: INFO)
- `CLASSIFICATION_THRESHOLD`: Threshold para classificação de candidatos (padrão: 0.25)

Veja a lista completa em [docs/env_variables.md](docs/env_variables.md).

## 🤖 Sistema Híbrido de Scoring + Clustering

### Pipeline Completo

Execute o pipeline completo com MLflow usando:

```bash
./scripts/run_pipeline.sh
```

Opções disponíveis:
- `--compare`: Treina e compara diferentes modelos
- `--port 8080`: Altera a porta do servidor MLflow (padrão: 5001)
- `--no-server`: Executa o pipeline sem iniciar o servidor MLflow
- `--no-cv`: Desativa a validação cruzada
- `--no-leakage-prevention`: Desativa a detecção de data leakage
- `--no-feature-selection`: Desativa a seleção de features
- `--cv-folds 10`: Altera o número de folds na validação cruzada (padrão: 5)

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

## 🚀 API de Scoring

O projeto inclui uma API para servir o modelo de Machine Learning treinado:

### Executando a API Localmente
```bash
./scripts/start_api.sh
```

### Utilizando Docker
```bash
docker build -t decision-scoring-api .
docker run -p 8000:8000 decision-scoring-api
```

### Endpoints Principais
- `POST /score` - Endpoint principal para predições individuais
- `POST /score/batch` - Processamento de múltiplos candidatos em lote
- `GET /health` - Health check da API
- `GET /metrics` - Métricas de desempenho da API (requer autenticação admin)
- `GET /monitoring/drift` - Análise de drift do modelo (requer autenticação admin)
- `GET /monitoring/drift/visualization` - Visualização de drift para uma feature específica
- `GET /monitoring/metrics/history` - Histórico de métricas do modelo
- `GET /monitoring/predictions/recent` - Estatísticas sobre predições recentes

### Sistema de Monitoramento

O projeto inclui um sistema completo de monitoramento para acompanhar o desempenho do modelo e detectar drift:

#### Dashboard de Métricas

Para iniciar o dashboard de monitoramento:

```bash
./scripts/start_dashboard.sh
```

O dashboard fornece:
- Visualização em tempo real das métricas do modelo
- Análise de drift entre dados de treino e produção
- Histórico de desempenho do modelo
- Estatísticas sobre predições recentes

#### Verificação de Drift

Para verificar drift e gerar alertas:

```bash
./scripts/check_drift.sh
```

Consulte a [documentação completa do sistema de monitoramento](docs/monitoring_guide.md) para mais detalhes.
- `/docs` - Documentação interativa (Swagger UI)

### Autenticação
Todas as requisições devem incluir um cabeçalho `X-API-Key` com uma chave válida:
- `your-api-key`: Acesso de administrador (todos os endpoints)
- `test-api-key`: Acesso somente leitura (endpoints básicos)

### Exemplo de Requisição
```bash
curl -X POST "http://localhost:8000/score" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "idade": 28,
    "experiencia": 5,
    "educacao": "ensino_superior",
    "area_formacao": "tecnologia",
    "vaga_titulo": "Desenvolvedor Python",
    "vaga_area": "tecnologia",
    "vaga_senioridade": "pleno"
  }'
```

### Exemplo de Resposta
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

### Implantação no Render
O projeto pode ser facilmente implantado na plataforma Render usando Docker ou o arquivo de configuração incluído.

#### Opções de Implantação
1. **Via Blueprint (config/render/render.yaml)**: Implantação automática usando nosso arquivo de configuração
2. **Via Docker**: Implantação manual do contêiner Docker usando o config/docker/api/Dockerfile incluído
3. **Sem Docker**: Implantação usando o ambiente Python do Render e o config/render/Procfile

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

## 🧠 Recursos de IA e LLM

O projeto agora inclui recursos de IA para gerar comentários personalizados sobre candidatos:

### Comentários LLM para Recomendações
A API agora gera automaticamente comentários em linguagem natural para cada recomendação de candidato. Este recurso:

- Analisa o perfil do candidato e os requisitos da vaga
- Gera texto explicativo sobre o motivo da recomendação positiva ou negativa
- Adapta o tom e conteúdo com base na probabilidade da predição
- Inclui detalhes relevantes como experiência e formação do candidato

### Como Funciona
O sistema utiliza:
1. **TextBlob** para processamento de linguagem natural
2. **Templates personalizados** para diferentes cenários de recomendação
3. **Lógica contextual** para selecionar os detalhes mais relevantes a destacar

Para mais detalhes sobre esta funcionalidade, consulte a documentação em [docs/llm_comments.md](docs/llm_comments.md)

## 🔮 Próximos Passos

1. **Experimentação com MLflow**: 
   - ✅ Otimização de hiperparâmetros - implementado com RandomizedSearchCV
   - ✅ Teste de diferentes algoritmos - implementado com comparação de modelos
   - ✅ Comparação de métricas de desempenho - tracking com MLflow

2. **Implementação de Clustering**:
   - Segmentação de candidatos por perfil
   - Identificação de grupos de vagas similares
   - Integração do clustering ao scoring model

3. **Melhorias no Modelo**:
   - Feature engineering avançado
   - Implementação de técnicas de deep learning
   - Validação cruzada para maior robustez

4. **Produtivização**:
   - ✅ API REST para servir o modelo - implementada com FastAPI
   - ✅ Implantação no Render - configurada com Docker
   - ✅ Comentários LLM para explicabilidade - implementado com TextBlob
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

## 🧪 Testes Unitários

O projeto conta com testes unitários para garantir a qualidade e o funcionamento correto dos componentes da pipeline. Para executar os testes:

```bash
# Execute o script de testes
./scripts/run_tests.sh
```

Para mais detalhes sobre os testes implementados, consulte o [README dos testes](tests/README.md).

## 📞 Contato

FIAP Tech Challenge 05 - [@rivolela](https://github.com/rivolela/fiap_tech_challenge_05)

---
**Nota**: Certifique-se de que os arquivos JSON estejam disponíveis na pasta `data/` antes de executar o notebook de análise.



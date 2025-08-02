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
├── data/
│   ├── applicants.json
│   ├── prospects.json
│   └── vagas.json
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

1. **Limpeza de Dados**: Tratar valores ausentes e outliers identificados
2. **Feature Engineering**: Criar novas variáveis baseadas nos insights
3. **Modelagem**: Aplicar algoritmos de machine learning para:
   - Previsão de sucesso de candidatos
   - Matching entre candidatos e vagas
   - Otimização do processo de recrutamento

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

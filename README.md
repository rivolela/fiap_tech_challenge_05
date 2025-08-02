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
├── dados/
│   ├── candidatos.csv
│   ├── entrevistas.csv
│   └── vagas.csv
├── requirements.txt
├── setup.sh
├── check_env.py
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

3. Abrir o notebook `notebooks/Análise Exploratória dos Dados.ipynb`

## 📝 Dados

Certifique-se de que os seguintes arquivos CSV estejam na pasta `dados/`:
- `candidatos.csv`: Dados dos candidatos
- `entrevistas.csv`: Dados das entrevistas
- `vagas.csv`: Dados das vagas

## 🤝 Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 📞 Contato

FIAP Tech Challenge 05 - [@rivolela](https://github.com/rivolela/fiap_tech_challenge_05) 

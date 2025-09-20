# API de Scoring Decision

Este componente fornece uma API REST para servir o modelo de scoring desenvolvido para a Decision.

## Funcionalidades

- Endpoint para predição de sucesso na contratação de candidatos
- Processamento de predições individuais ou em lote
- Comentários personalizados via LLM sobre cada recomendação
- Suporte para informações de vagas de emprego
- Autenticação via API key
- Monitoramento de saúde e métricas de desempenho
- Documentação interativa com Swagger UI

## Requisitos

- Python 3.8+
- FastAPI
- Uvicorn
- Pandas e NumPy
- Scikit-learn

## Executando Localmente

### 1. Treinando o modelo

Antes de iniciar a API, certifique-se de ter o modelo treinado:

```bash
# Execute o pipeline de treinamento
./run_pipeline.sh
```

### 2. Iniciando a API

```bash
# Usando o script de inicialização
./start_api.sh

# Ou manualmente
export PYTHONPATH=$(pwd)
uvicorn src.api.scoring_api:app --host 0.0.0.0 --port 8000 --reload
```

A API estará disponível em: http://localhost:8000  
A documentação interativa estará em: http://localhost:8000/docs

## Usando Docker

### Construir a imagem

```bash
docker build -t decision-scoring-api .
```

### Executar o contêiner

```bash
docker run -p 8000:8000 decision-scoring-api
```

## Endpoints da API

### 1. Predição Individual

**Endpoint**: `/predict/`  
**Método**: POST  
**Autenticação**: Requerida (api_key)

**Exemplo de Requisição**:
```json
{
  "idade": 32,
  "experiencia": 6,
  "educacao": "ensino_superior",
  "area_formacao": "tecnologia",
  "habilidades": ["python", "machine_learning", "estatistica"],
  "vaga_titulo": "Desenvolvedor Python",
  "vaga_area": "tecnologia",
  "vaga_senioridade": "pleno",
  "vaga_id": "vaga-123"
}
```

**Exemplo de Resposta**:
```json
{
  "prediction": 1,
  "probability": 0.85,
  "recommendation": "Recomendado",
  "comment": "A análise indica boa adequação ao cargo de Desenvolvedor Python na área de tecnologia, nível pleno. O candidato possui 6.0 anos de experiência relevante e formação superior na área de tecnologia.",
  "vaga_info": {
    "id": "vaga-123",
    "titulo": "Desenvolvedor Python",
    "area": "tecnologia",
    "senioridade": "pleno"
  },
  "match_score": 0.78
}
```

### 2. Predição em Lote

**Endpoint**: `/predict/batch/`  
**Método**: POST  
**Autenticação**: Requerida (api_key)

**Exemplo de Requisição**:
```json
{
  "candidates": [
    {
      "idade": 32,
      "experiencia": 6,
      "educacao": "ensino_superior",
      "habilidades": ["python", "machine_learning", "estatistica"]
    },
    {
      "idade": 26,
      "experiencia": 2,
      "educacao": "ensino_medio",
      "habilidades": ["excel", "atendimento"]
    }
  ]
}
```

### 3. Verificação de Saúde

**Endpoint**: `/health/`  
**Método**: GET  
**Autenticação**: Não requerida

### 4. Métricas de Desempenho

**Endpoint**: `/metrics/`  
**Método**: GET  
**Autenticação**: Requerida (api_key com permissão admin)

## Autenticação

A API utiliza uma autenticação simples via API key. Você pode fornecer a chave de duas maneiras:

### 1. Via Query String (parâmetro na URL)

```
/predict/?api_key=your-api-key
```

### 2. Via Cabeçalho HTTP

```
X-API-Key: your-api-key
```

As chaves de API disponíveis para teste são:
- `your-api-key`: Permissão de administrador (acesso a todas as funcionalidades)
- `test-api-key`: Permissão somente leitura (acesso limitado)

## Integração com Sistemas Existentes

### Python

```python
import requests
import json

API_URL = "http://localhost:8000/predict/"
API_KEY = "your-api-key"

dados_candidato = {
    "idade": 32,
    "experiencia": 6,
    "educacao": "ensino_superior",
    "habilidades": ["python", "machine_learning", "estatistica"]
}

response = requests.post(
    API_URL,
    headers={"X-API-Key": API_KEY},  # Autenticação via cabeçalho
    json=dados_candidato
)

# Ou, alternativamente, usando query string:
# response = requests.post(
#     f"{API_URL}?api_key={API_KEY}",
#     json=dados_candidato
# )

if response.status_code == 200:
    resultado = response.json()
    print(f"Recomendação: {resultado['recommendation']}")
    print(f"Probabilidade: {resultado['probability'] * 100:.1f}%")
else:
    print(f"Erro: {response.status_code} - {response.text}")
```

### JavaScript

```javascript
const apiUrl = "http://localhost:8000/predict/";
const apiKey = "your-api-key";

const candidateData = {
  idade: 32,
  experiencia: 6,
  educacao: "ensino_superior",
  habilidades: ["python", "machine_learning", "estatistica"]
};

fetch(apiUrl, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': apiKey  // Autenticação via cabeçalho
  },
  body: JSON.stringify(candidateData)
  
// Ou, alternativamente, usando query string:
// fetch(`${apiUrl}?api_key=${apiKey}`, {
//   method: 'POST',
//   headers: {
//     'Content-Type': 'application/json'
//   },
//   body: JSON.stringify(candidateData)
})
.then(response => response.json())
.then(data => {
  console.log(`Recomendação: ${data.recommendation}`);
  console.log(`Probabilidade: ${(data.probability * 100).toFixed(1)}%`);
})
.catch(error => console.error('Erro:', error));
```

## Produção

Para ambientes de produção, recomenda-se:

1. Usar um servidor proxy como Nginx
2. Implementar HTTPS
3. Usar um sistema mais seguro para gerenciar chaves de API
4. Configurar monitoramento e alertas
5. Implementar registro de logs centralizado

### Exemplo de Configuração Nginx

Veja o arquivo `nginx.conf.example` para uma configuração básica do Nginx como proxy reverso para a API.

## Desenvolvido por

Decision Tech Team
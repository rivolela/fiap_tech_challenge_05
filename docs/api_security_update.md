# Documentação de Atualização de Segurança da API

## Mudança de Autenticação API Key

A partir de 29/09/2025, a API foi atualizada para aceitar somente autenticação via **header HTTP X-API-Key**.

### Mudanças Implementadas

1. **Remoção do método de query parameter para API Keys**:
   - Antes: `?api_key=your-api-key` ou `X-API-Key: your-api-key` 
   - Agora: **Somente** `X-API-Key: your-api-key`

2. **Documentação Swagger atualizada** para refletir o novo método de autenticação.

3. **Logs de segurança aprimorados** para reduzir exposição de informações sensíveis.

### Motivos da Mudança

Esta atualização foi implementada por motivos de segurança:

1. **Exposição de credenciais**: API Keys em URLs podem ser:
   - Registradas em logs de servidor
   - Armazenadas em históricos de navegadores
   - Visíveis em referências HTTP
   - Salvas em caches e proxies

2. **Boas práticas de segurança**: Conformidade com padrões modernos de API REST e segurança.

### Como Atualizar sua Integração

Se você estiver enviando a API Key como parâmetro de query, atualize seu código para enviá-la como header HTTP:

```python
# Antes
requests.post('http://api.example.com/predict?api_key=fiap-api-key', json=data)

# Agora
headers = {'X-API-Key': 'fiap-api-key'}
requests.post('http://api.example.com/predict', headers=headers, json=data)
```

```javascript
// Antes
fetch('http://api.example.com/predict?api_key=fiap-api-key', {
  method: 'POST',
  body: JSON.stringify(data)
})

// Agora
fetch('http://api.example.com/predict', {
  method: 'POST',
  headers: {
    'X-API-Key': 'fiap-api-key',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(data)
})
```

### Exemplos de Uso

#### cURL:
```bash
curl -X POST 'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'X-API-Key: fiap-api-key' \
  -H 'Content-Type: application/json' \
  -d '{"idade": 26, "experiencia": 3.9, "educacao": "ensino_superior", ...}'
```
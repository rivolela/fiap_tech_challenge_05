# Checklist de Deployment para Render

Use esta checklist para garantir uma implantação bem-sucedida no Render:

## Antes do Deployment

- [x] Configurar o arquivo `render.yaml` com as configurações corretas
- [x] Certificar-se de que o `Dockerfile` está funcionando corretamente
- [x] Criar um arquivo `Procfile` (caso deseje usar implantação sem Docker)
- [x] Garantir que `requirements.txt` inclui todas as dependências necessárias
- [x] Certificar-se de que `gunicorn` está incluído nas dependências
- [x] Implementar endpoint de health check (`/health`)
- [x] Testar a API localmente para garantir que ela funciona corretamente
- [ ] Verificar se todos os modelos necessários estão incluídos no repositório
- [ ] Garantir que o repositório está atualizado no GitHub

## Durante o Deployment no Render

- [ ] Criar uma conta no [Render](https://render.com/) caso ainda não tenha
- [ ] Conectar sua conta do GitHub ao Render
- [ ] Selecionar o repositório `fiap_tech_challenge_05`
- [ ] Escolher o método de implantação:
  - [ ] Opção 1: Blueprint (usando render.yaml)
  - [ ] Opção 2: Web Service via Docker
  - [ ] Opção 3: Web Service via Python
- [ ] Verificar se todas as variáveis de ambiente estão configuradas
- [ ] Iniciar o deployment e aguardar a conclusão

## Após o Deployment

- [ ] Verificar se a API está online acessando o URL fornecido pelo Render
- [ ] Testar o endpoint de health check: `https://sua-api-url.onrender.com/health`
- [ ] Verificar a documentação da API: `https://sua-api-url.onrender.com/docs`
- [ ] Testar a autenticação com uma API key válida
- [ ] Realizar uma predição de teste usando o endpoint `/score`
- [ ] Monitorar os logs para verificar se há erros
- [ ] Configurar alertas para monitoramento (opcional)

## Solução de Problemas Comuns

1. **API não inicia**:
   - Verifique os logs no dashboard do Render
   - Confirme se todas as dependências estão em `requirements.txt`
   - Teste o Dockerfile localmente

2. **Erro de módulo não encontrado**:
   - Verifique se a variável de ambiente `PYTHONPATH` está configurada corretamente
   - Certifique-se de que a estrutura de arquivos está correta

3. **Problemas de autenticação**:
   - Verifique se está enviando o cabeçalho `X-API-Key` corretamente
   - Confirme se a API key está na lista de chaves válidas

4. **Memória insuficiente**:
   - Considere usar um plano pago do Render com mais recursos
   - Otimize o carregamento do modelo para usar menos memória
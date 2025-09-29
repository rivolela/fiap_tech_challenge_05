#!/bin/bash

# Script para realizar commit e push das correções dos testes

# Adicionar os arquivos modificados
git add requirements.txt tests/unit/api/test_security.py tests/unit/features/test_data_validation.py

# Criar commit com mensagem descritiva
git commit -m "Corrige testes assíncronos e problema de diretório inexistente

- Adiciona pytest-asyncio para suportar funções assíncronas nos testes
- Adiciona decorador @pytest.mark.asyncio aos testes de segurança
- Corrige o mock de os.path.exists para criar o diretório data/insights durante os testes"

# Enviar alterações para o repositório remoto
git push origin main

echo "Commit e push realizados com sucesso!"
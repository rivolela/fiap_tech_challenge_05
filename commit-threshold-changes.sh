#!/bin/bash

# Script para fazer commit das alterações do threshold

# Adicionar arquivos modificados
git add src/api/model_loader.py README.md docker-compose.yml

# Criar commit com mensagem descritiva
git commit -m "Ajuste do threshold de classificação de 0.25 para 0.5 para melhorar a consistência entre a probabilidade, recomendação e comentários"

# Exibir status após o commit
git status

echo "Commit realizado com sucesso!"
#!/bin/bash

# Script para fazer commit das alterações e enviar para o repositório remoto

# Adicionar arquivos modificados
git add README.md

# Criar commit com mensagem descritiva
git commit -m "Adiciona URL da API em produção ao README"

# Enviar alterações para o repositório remoto
git push origin main

echo "Commit e push realizados com sucesso!"
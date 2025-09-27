#!/bin/bash
# sync_dockerfile.sh - Sincroniza o Dockerfile da raiz com o original em config/docker/api

# Certifica-se de que estamos na raiz do projeto
cd "$(dirname "$0")/.."

# Verifica se o Dockerfile original existe
if [ ! -f "config/docker/api/Dockerfile" ]; then
    echo "Erro: Dockerfile original não encontrado em config/docker/api/Dockerfile"
    exit 1
fi

# Copia o Dockerfile original para a raiz
echo "Copiando Dockerfile de config/docker/api para a raiz do projeto..."
cp config/docker/api/Dockerfile ./Dockerfile

# Verifica se a cópia foi bem sucedida
if [ $? -eq 0 ]; then
    echo "Dockerfile sincronizado com sucesso!"
else
    echo "Erro ao copiar o Dockerfile"
    exit 1
fi

echo "Lembre-se: O Dockerfile na raiz é apenas uma cópia para compatibilidade com o deploy."
echo "Sempre faça alterações no arquivo original em config/docker/api/Dockerfile primeiro!"
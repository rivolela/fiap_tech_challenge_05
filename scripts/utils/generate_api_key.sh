#!/bin/bash

# Script para gerar API keys para o sistema

# Função para hashear a API key
hash_api_key() {
  local api_key=$1
  local salt=$2
  
  # Usar OpenSSL para gerar o hash (HMAC-SHA256)
  echo -n "$api_key" | openssl dgst -sha256 -hmac "$salt" | awk '{print $NF}'
}

# Verificar se o salt foi fornecido
if [ -z "$1" ]; then
  echo "Uso: $0 <salt>"
  echo "O salt deve ser o mesmo configurado na variável API_KEY_SALT no arquivo .env"
  exit 1
fi

salt=$1

# Ler a API key do usuário
read -p "Digite a API key que deseja hashear: " api_key

# Hashear a API key
hashed_key=$(hash_api_key "$api_key" "$salt")

echo ""
echo "API Key Original: $api_key"
echo "API Key Hasheada: $hashed_key"
echo ""
echo "Para usar esta API key no código, adicione ao dicionário API_KEYS em security.py:"
echo "\"$hashed_key\": \"role\","
echo ""
echo "Ou adicione ao formato JSON nas variáveis de ambiente:"
echo "{\"key\": \"$api_key\", \"role\": \"admin\"},"
echo ""
#!/bin/bash

echo "Testando API após correção..."
echo "Esperando API iniciar..."
sleep 5

echo "Fazendo requisição com o exemplo do caso problemático..."
curl -X POST 'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'X-API-Key: fiap-api-key' \
  -H 'Content-Type: application/json' \
  -d '{"idade": 26, "experiencia": 3.9, "educacao": "ensino_superior", "tempo_desempregado": 1.4, "area_formacao": "engenharia", "habilidades": ["comunicação", "trabalho em equipe"], "cargo_anterior": "Especialista de Geral", "salario_anterior": 10026, "anos_estudo": 16, "vaga_id": "1715", "vaga_titulo": "SAP BPC - 20191623853", "vaga_area": "geral", "vaga_senioridade": "pleno", "extra_data": {"situacao_original": "Contratado pela Decision", "data_candidatura": "09-01-2020", "nome_candidato": "Davi"}}'

echo ""
echo "Feito!"
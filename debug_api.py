# Script para diagnóstico da API
import pandas as pd

# Simulando dados de entrada
data = pd.DataFrame([{
  "idade": 28,
  "experiencia": 5.5,
  "educacao": "ensino_superior",
  "tempo_desempregado": 0.2,
  "area_formacao": "tecnologia",
  "habilidades": [
    "python", "javascript", "react", "machine learning", "docker"
  ],
  "cargo_anterior": "Desenvolvedor Full Stack",
  "salario_anterior": 8500,
  "anos_estudo": 16
}])

# Simulando informações da vaga
vaga_info = {
  "id": "tech-123",
  "titulo": "Desenvolvedor Full Stack Senior",
  "area": "tecnologia",
  "senioridade": "senior"
}

# Diagnóstico de valores
print("--- DIAGNÓSTICO DE VALORES ---")
print(f"area_formacao: '{data['area_formacao'].iloc[0]}', tipo: {type(data['area_formacao'].iloc[0])}")
print(f"vaga_area: '{vaga_info['area']}', tipo: {type(vaga_info['area'])}")

# Teste de comparação
match_simples = data['area_formacao'].iloc[0].lower() == vaga_info['area'].lower()
print(f"Comparação simples: {match_simples}")

# Teste condições
print(f"area_formacao existe? {bool(data['area_formacao'].iloc[0])}")
print(f"area_formacao != 'string'? {data['area_formacao'].iloc[0] != 'string'}")
print(f"area_formacao.lower() != 'none'? {data['area_formacao'].iloc[0].lower() != 'none'}")
print(f"vaga_area existe? {bool(vaga_info['area'])}")
print(f"vaga_area != 'string'? {vaga_info['area'] != 'string'}")
print(f"vaga_area.lower() != 'none'? {vaga_info['area'].lower() != 'none'}")

# Teste completo
match_area = False
if (data['area_formacao'].iloc[0] and data['area_formacao'].iloc[0] != "string" and data['area_formacao'].iloc[0].lower() != "none" and 
    vaga_info['area'] and vaga_info['area'] != "string" and vaga_info['area'].lower() != "none"):
    match_area = (data['area_formacao'].iloc[0].lower() == vaga_info['area'].lower() or 
                  data['area_formacao'].iloc[0].lower() in vaga_info['area'].lower() or 
                  vaga_info['area'].lower() in data['area_formacao'].iloc[0].lower())
    print(f"Condição completa satisfeita, resultado: {match_area}")
else:
    print("Condição completa NÃO satisfeita")

#!/usr/bin/env python3
"""Script para gerar exemplos de comandos curl baseados em candidatos aprovados e reprovados do arquivo prospects.json"""

import json
import random
import os

# Situações que indicam aprovação e reprovação
APPROVED_STATUSES = [
    "Contratado pela Decision",
    "Contratado como Hunting",
    "Aprovado"
]

REJECTED_STATUSES = [
    "Não Aprovado pelo Cliente",
    "Não Aprovado pelo RH",
    "Não Aprovado pelo Requisitante",
    "Recusado"
]

def load_prospects(filename):
    """Carrega os dados de prospects do arquivo JSON"""
    print(f"Carregando dados de {filename}...")
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data

def extract_candidates_by_status(prospects_data, status_list, max_samples=5):
    """Extrai candidatos com os status especificados"""
    candidates = []
    
    for job_id, job_data in prospects_data.items():
        job_title = job_data.get('titulo', 'Desconhecido')
        
        for prospect in job_data.get('prospects', []):
            if prospect.get('situacao_candidado') in status_list:
                candidates.append({
                    'job_id': job_id,
                    'job_title': job_title,
                    'candidate': prospect
                })
    
    # Limitar o número de amostras
    if len(candidates) > max_samples:
        candidates = random.sample(candidates, max_samples)
    
    return candidates

def generate_api_data(candidate_info):
    """Gera dados para a API com base nas informações do candidato"""
    job_id = candidate_info['job_id']
    job_title = candidate_info['job_title']
    candidate = candidate_info['candidate']
    
    # Extrair área da vaga a partir do título
    job_area = "geral"
    job_title_lower = job_title.lower()
    
    if any(term in job_title_lower for term in ["desenvolvedor", "programador", "software", "ti ", "analista de sistemas", "devops", "fullstack"]):
        job_area = "tecnologia"
    elif any(term in job_title_lower for term in ["vendas", "comercial", "sales"]):
        job_area = "comercial"
    elif any(term in job_title_lower for term in ["marketing", "digital", "mídias"]):
        job_area = "marketing"
    elif any(term in job_title_lower for term in ["financeiro", "contábil", "fiscal"]):
        job_area = "financeiro"
    elif any(term in job_title_lower for term in ["recursos humanos", "rh", "people"]):
        job_area = "rh"
    
    # Extrair senioridade a partir do título
    job_seniority = "pleno"  # valor padrão
    
    if any(term in job_title_lower for term in ["senior", "sênior", "sr", "especialista"]):
        job_seniority = "senior"
    elif any(term in job_title_lower for term in ["junior", "júnior", "jr", "trainee"]):
        job_seniority = "junior"
    elif any(term in job_title_lower for term in ["pleno", "pl"]):
        job_seniority = "pleno"
    
    # Gerar idade aleatória (mas realista) com base no tipo de vaga
    base_age = 25
    if job_seniority == "senior":
        base_age = 35
    elif job_seniority == "junior":
        base_age = 23
    
    age = base_age + random.randint(-3, 5)
    
    # Gerar experiência com base na senioridade
    if job_seniority == "senior":
        experience = random.uniform(6.0, 15.0)
    elif job_seniority == "pleno":
        experience = random.uniform(3.0, 6.0)
    else:
        experience = random.uniform(0.5, 3.0)
    
    # Definir área de formação com chance de ser compatível com área da vaga
    match_rate = 0.7  # 70% de chance de a formação ser compatível com a vaga
    
    formation_area = job_area if random.random() < match_rate else random.choice(
        ["tecnologia", "comercial", "marketing", "financeiro", "rh", "administracao", "engenharia"]
    )
    
    # Definir educação com base na senioridade
    education_levels = {
        "junior": ["ensino_superior", "ensino_medio"],
        "pleno": ["ensino_superior", "pos_graduacao"],
        "senior": ["ensino_superior", "pos_graduacao"]
    }
    
    education = random.choice(education_levels.get(job_seniority, ["ensino_superior"]))
    
    # Gerar habilidades relevantes com base na área
    skills_by_area = {
        "tecnologia": [
            "python", "java", "javascript", "sql", "aws", "docker", "kubernetes",
            "angular", "react", "node.js", "machine learning", "data science"
        ],
        "comercial": [
            "vendas", "negociação", "prospecção", "gestão de clientes", "crm",
            "estratégia comercial", "inside sales", "vendas corporativas"
        ],
        "marketing": [
            "marketing digital", "redes sociais", "seo", "google analytics",
            "inbound marketing", "outbound marketing", "copywriting", "branding"
        ],
        "financeiro": [
            "contabilidade", "fiscal", "gestão financeira", "análise de custos",
            "controladoria", "planejamento financeiro", "auditoria"
        ],
        "rh": [
            "recrutamento", "seleção", "treinamento", "departamento pessoal",
            "benefícios", "folha de pagamento", "clima organizacional"
        ]
    }
    
    # Selecionar 3-6 habilidades relevantes
    available_skills = skills_by_area.get(job_area, ["comunicação", "trabalho em equipe"])
    num_skills = random.randint(3, 6)
    skills = random.sample(available_skills, min(num_skills, len(available_skills)))
    
    # Dados do candidato para a API
    api_data = {
        "idade": age,
        "experiencia": round(experience, 1),
        "educacao": education,
        "tempo_desempregado": round(random.uniform(0, 1.5), 1),
        "area_formacao": formation_area,
        "habilidades": skills,
        "cargo_anterior": f"{random.choice(['Analista', 'Especialista', 'Consultor', 'Gerente'])} de {job_area.title()}",
        "salario_anterior": random.randint(3000, 18000),
        "anos_estudo": 12 if education == "ensino_medio" else 16 if education == "ensino_superior" else 18,
        "vaga_id": job_id,
        "vaga_titulo": job_title,
        "vaga_area": job_area,
        "vaga_senioridade": job_seniority,
        "extra_data": {
            "situacao_original": candidate.get("situacao_candidado", ""),
            "data_candidatura": candidate.get("data_candidatura", ""),
            "nome_candidato": candidate.get("nome", "").split()[0]  # Apenas primeiro nome para privacidade
        }
    }
    
    return api_data

def generate_curl_command(api_data, api_key="fiap-api-key"):
    """Gera o comando curl para testar a API com os dados fornecidos"""
    data_json = json.dumps(api_data, ensure_ascii=False)
    curl_cmd = f'''curl -X POST 'http://localhost:8000/predict?api_key={api_key}' \\
  -H 'accept: application/json' \\
  -H 'Content-Type: application/json' \\
  -d '{data_json}' '''
    return curl_cmd

def main():
    """Função principal"""
    try:
        prospects_file = "data/raw/prospects.json"
        if not os.path.exists(prospects_file):
            print(f"Arquivo {prospects_file} não encontrado!")
            return
        
        # Criar diretório para os exemplos curl se não existir
        os.makedirs("examples", exist_ok=True)
        
        # Carregar dados de prospects
        prospects_data = load_prospects(prospects_file)
        
        # Extrair candidatos aprovados e reprovados
        approved_candidates = extract_candidates_by_status(prospects_data, APPROVED_STATUSES, max_samples=5)
        rejected_candidates = extract_candidates_by_status(prospects_data, REJECTED_STATUSES, max_samples=5)
        
        print(f"Encontrados {len(approved_candidates)} candidatos aprovados e {len(rejected_candidates)} candidatos reprovados")
        
        # Gerar comandos curl e dados da API
        approved_examples = []
        for candidate_info in approved_candidates:
            api_data = generate_api_data(candidate_info)
            curl_cmd = generate_curl_command(api_data)
            
            # Adicionar status original aos dados
            status = candidate_info['candidate'].get('situacao_candidado', '')
            
            approved_examples.append({
                "status": status,
                "job_title": candidate_info['job_title'],
                "command": curl_cmd,
                "data": api_data
            })
        
        rejected_examples = []
        for candidate_info in rejected_candidates:
            api_data = generate_api_data(candidate_info)
            curl_cmd = generate_curl_command(api_data)
            
            # Adicionar status original aos dados
            status = candidate_info['candidate'].get('situacao_candidado', '')
            
            rejected_examples.append({
                "status": status,
                "job_title": candidate_info['job_title'],
                "command": curl_cmd,
                "data": api_data
            })
        
        # Salvar comandos curl em arquivos separados
        with open("examples/approved_candidates_curl.txt", "w") as f:
            f.write("# Comandos curl para candidatos APROVADOS\n\n")
            for i, example in enumerate(approved_examples):
                f.write(f"\n\n# Exemplo {i+1} - {example['status']} - {example['job_title']}\n")
                f.write(f"{example['command']}\n")
        
        with open("examples/rejected_candidates_curl.txt", "w") as f:
            f.write("# Comandos curl para candidatos REPROVADOS\n\n")
            for i, example in enumerate(rejected_examples):
                f.write(f"\n\n# Exemplo {i+1} - {example['status']} - {example['job_title']}\n")
                f.write(f"{example['command']}\n")
        
        # Salvar dados de API em arquivos JSON para referência
        with open("examples/approved_candidates_data.json", "w") as f:
            json.dump([example["data"] for example in approved_examples], f, indent=2, ensure_ascii=False)
        
        with open("examples/rejected_candidates_data.json", "w") as f:
            json.dump([example["data"] for example in rejected_examples], f, indent=2, ensure_ascii=False)
        
        print("Arquivos gerados:")
        print("- examples/approved_candidates_curl.txt - Comandos curl para candidatos aprovados")
        print("- examples/rejected_candidates_curl.txt - Comandos curl para candidatos reprovados")
        print("- examples/approved_candidates_data.json - Dados JSON para candidatos aprovados")
        print("- examples/rejected_candidates_data.json - Dados JSON para candidatos reprovados")
        
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()

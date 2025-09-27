#!/usr/bin/env python3
"""
Script para reorganizar a estrutura do projeto Decision Scoring.
Este script implementa o plano de reorganização definido para melhorar
a organização do projeto, movendo arquivos para locais mais apropriados
e atualizando referências.

Uso:
    python reorganize_project.py [--dry-run] [--backup]

Opções:
    --dry-run   Apenas mostra o que seria feito, sem fazer alterações
    --backup    Cria um backup do projeto antes de fazer alterações
"""

import os
import sys
import shutil
import argparse
import glob
import re
from datetime import datetime
from pathlib import Path

# Configuração
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BACKUP_DIR = PROJECT_ROOT / "backup"

# Define o plano de movimentação de arquivos
FILE_MOVEMENT_PLAN = {
    # Arquivos Docker
    'config/docker/api/Dockerfile': 'config/docker/api/config/docker/api/Dockerfile',
    'config/docker/api/config/docker/dashboard/Dockerfile': 'config/docker/dashboard/config/docker/api/Dockerfile',
    'config/docker/docker-compose.yml': 'config/docker/config/docker/docker-compose.yml',
    
    # Configuração Nginx
    'config/nginx/nginx.conf': 'config/nginx/nginx.conf',
    
    # Configuração Render
    'config/render/render.yaml': 'config/render/config/render/render.yaml',
    'config/render/Procfile': 'config/render/config/render/Procfile',
    'config/render/.env.render': 'config/render/config/render/.env.render',
    
    # Scripts
    'scripts/utils/debug_api.py': 'scripts/utils/scripts/utils/debug_api.py',
    'scripts/deployment/quick_deploy.sh': 'scripts/deployment/scripts/deployment/quick_deploy.sh',
    'scripts/utils/commit-changes.sh': 'scripts/utils/scripts/utils/commit-changes.sh',
    'scripts/utils/generate_prospects_curl.py': 'scripts/utils/scripts/utils/generate_prospects_curl.py',
    
    # Logs
    'api_logs.log': 'logs/api_logs.log',
}

# Arquivos a serem removidos (temporários, etc.)
FILES_TO_REMOVE = [
    'test_report.html',
]

# Estrutura de diretórios a ser criada
DIRECTORY_STRUCTURE = [
    '.github/workflows',
    'config/docker/api',
    'config/docker/dashboard',
    'config/nginx',
    'config/render',
    'logs',
    'scripts/deployment',
    'scripts/monitoring',
    'scripts/utils',
]

def parse_args():
    parser = argparse.ArgumentParser(description='Reorganiza a estrutura do projeto')
    parser.add_argument('--dry-run', action='store_true', help='Não faz alterações reais')
    parser.add_argument('--backup', action='store_true', help='Cria backup antes das alterações')
    return parser.parse_args()

def create_backup(dry_run=False):
    """Cria um backup do projeto."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = BACKUP_DIR / f"backup_{timestamp}"
    
    print(f"Criando backup em {backup_path}")
    if not dry_run:
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copiar todos os arquivos exceto pastas grandes/desnecessárias
        for item in PROJECT_ROOT.glob('*'):
            if item.name in ['.git', '.venv', 'backup', '__pycache__']:
                continue
                
            if item.is_dir():
                shutil.copytree(item, backup_path / item.name)
            else:
                shutil.copy2(item, backup_path)
    
    return backup_path

def create_directory_structure(dry_run=False):
    """Cria a estrutura de diretórios necessária."""
    print("Criando estrutura de diretórios...")
    
    for dir_path in DIRECTORY_STRUCTURE:
        full_path = PROJECT_ROOT / dir_path
        print(f"  Criando diretório: {dir_path}")
        
        if not dry_run:
            full_path.mkdir(parents=True, exist_ok=True)

def move_files(dry_run=False):
    """Move os arquivos conforme o plano de movimentação."""
    print("Movendo arquivos...")
    
    for src_file, dest_path in FILE_MOVEMENT_PLAN.items():
        src_path = PROJECT_ROOT / src_file
        dest_full_path = PROJECT_ROOT / dest_path
        
        if not src_path.exists():
            print(f"  AVISO: Arquivo de origem não encontrado: {src_file}")
            continue
        
        print(f"  Movendo: {src_file} -> {dest_path}")
        
        if not dry_run:
            # Garantir que o diretório de destino exista
            dest_full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Mover o arquivo
            shutil.move(src_path, dest_full_path)

def remove_files(dry_run=False):
    """Remove arquivos temporários ou desnecessários."""
    print("Removendo arquivos temporários...")
    
    for pattern in FILES_TO_REMOVE:
        for file_path in PROJECT_ROOT.glob(pattern):
            print(f"  Removendo: {file_path.relative_to(PROJECT_ROOT)}")
            
            if not dry_run:
                file_path.unlink()

def update_references(dry_run=False):
    """Atualiza referências aos arquivos movidos em outros arquivos."""
    print("Atualizando referências nos arquivos...")
    
    # Lista de diretórios onde procurar referências
    search_dirs = ['src', 'scripts', 'tests', 'notebooks']
    
    # Para cada arquivo movido, procura referências
    for src_file, dest_path in FILE_MOVEMENT_PLAN.items():
        # Pula alguns arquivos que não precisam ter referências atualizadas
        if src_file.endswith(('.log')):
            continue
            
        print(f"  Procurando referências a: {src_file}")
        
        # Constrói os padrões de busca (caminhos relativos e absolutos)
        search_patterns = [
            src_file,
            f"./{src_file}",
            str(PROJECT_ROOT / src_file)
        ]
        
        for search_dir in search_dirs:
            dir_path = PROJECT_ROOT / search_dir
            
            if not dir_path.exists():
                continue
                
            # Procura em todos os arquivos de código
            for ext in ['.py', '.sh', '.md', '.yml', '.yaml', '.json']:
                for code_file in dir_path.glob(f"**/*{ext}"):
                    try:
                        # Lê o conteúdo do arquivo
                        content = code_file.read_text()
                        updated = False
                        
                        # Verifica se há referências ao arquivo
                        for pattern in search_patterns:
                            if pattern in content:
                                print(f"    Encontrada referência em: {code_file.relative_to(PROJECT_ROOT)}")
                                
                                # Substitui o caminho antigo pelo novo
                                updated_content = content.replace(pattern, dest_path)
                                
                                if not dry_run and updated_content != content:
                                    code_file.write_text(updated_content)
                                    updated = True
                        
                        if updated:
                            print(f"    Atualizado: {code_file.relative_to(PROJECT_ROOT)}")
                    except Exception as e:
                        print(f"    ERRO ao processar {code_file}: {e}")

def update_readme(dry_run=False):
    """Atualiza o README.md com a nova estrutura do projeto."""
    readme_path = PROJECT_ROOT / "README.md"
    
    if not readme_path.exists():
        print("AVISO: README.md não encontrado")
        return
    
    print("Atualizando README.md...")
    
    try:
        readme_content = readme_path.read_text()
        
        # Seção para adicionar/atualizar
        structure_section = """
## Estrutura do Projeto

```
fiap_tech_challenge_05/
├── .github/              # GitHub Actions e templates para issues/PRs
├── config/               # Arquivos de configuração
│   ├── docker/           # Arquivos Docker
│   ├── nginx/            # Configuração Nginx
│   └── render/           # Configuração da plataforma Render
├── data/                 # Dados do projeto
├── docs/                 # Documentação
├── examples/             # Exemplos de uso
├── logs/                 # Logs da aplicação
├── models/               # Modelos treinados
├── notebooks/            # Jupyter notebooks
├── scripts/              # Scripts utilitários
│   ├── deployment/       # Scripts de implantação
│   ├── monitoring/       # Scripts de monitoramento
│   └── utils/            # Scripts utilitários diversos
├── src/                  # Código-fonte principal
└── tests/                # Testes
```
"""
        
        # Procura por uma seção existente sobre a estrutura do projeto
        structure_pattern = re.compile(r'#+\s*Estrutura\s+do\s+Projeto.*?(?=#+|$)', re.DOTALL)
        match = structure_pattern.search(readme_content)
        
        if match:
            # Atualiza a seção existente
            updated_readme = readme_content[:match.start()] + "## Estrutura do Projeto" + structure_section + readme_content[match.end():]
        else:
            # Adiciona uma nova seção no final
            updated_readme = readme_content + "\n\n" + "## Estrutura do Projeto" + structure_section
        
        # Atualiza também referências a caminhos
        for src_file, dest_path in FILE_MOVEMENT_PLAN.items():
            updated_readme = updated_readme.replace(src_file, dest_path)
        
        if not dry_run:
            readme_path.write_text(updated_readme)
            print("  README.md atualizado com sucesso")
    
    except Exception as e:
        print(f"ERRO ao atualizar README.md: {e}")

def create_contributing_file(dry_run=False):
    """Cria um arquivo CONTRIBUTING.md com diretrizes para contribuição."""
    contributing_path = PROJECT_ROOT / "CONTRIBUTING.md"
    
    # Verifica se o arquivo já existe
    if contributing_path.exists():
        print("AVISO: CONTRIBUTING.md já existe, pulando criação")
        return
    
    print("Criando CONTRIBUTING.md...")
    
    contributing_content = """# Contribuindo para o Projeto Decision Scoring

Agradecemos seu interesse em contribuir para o projeto! Aqui estão algumas diretrizes para ajudar você a começar.

## Fluxo de Trabalho

1. Faça fork do repositório
2. Clone o fork para sua máquina local
3. Configure o repositório upstream
4. Crie um branch para suas alterações
5. Faça suas alterações
6. Envie suas alterações para seu fork
7. Crie um Pull Request

## Convenções de Código

- Use snake_case para nomes de variáveis e funções
- Use PascalCase para nomes de classes
- Use UPPER_CASE para constantes
- Siga o padrão PEP 8 para código Python

## Estrutura do Projeto

Respeite a estrutura de diretórios do projeto:

- Código-fonte principal vai em `src/`
- Scripts utilitários vão em `scripts/`
- Configurações vão em `config/`
- Arquivos de log vão em `logs/`

## Commits

Use mensagens de commit claras e descritivas, seguindo o padrão:

```
tipo: descrição concisa

Descrição mais detalhada se necessário.
```

Onde `tipo` pode ser:
- `feat`: Nova funcionalidade
- `fix`: Correção de bug
- `docs`: Alterações na documentação
- `style`: Formatação, ponto e vírgula, etc; sem alteração de código
- `refactor`: Refatoração de código
- `test`: Adicionando testes, refatorando testes
- `chore`: Atualizações de tarefas de build, configurações, etc

## Testes

Certifique-se de adicionar testes para qualquer funcionalidade nova ou corrigida.
Execute a suite de testes antes de enviar seu Pull Request.

```bash
./scripts/run_tests.sh
```

## Documentação

Atualize a documentação relevante para suas alterações:

- README.md para alterações de uso
- Docstrings para funções e classes
- Comentários para código complexo
"""

    if not dry_run:
        contributing_path.write_text(contributing_content)
        print("  CONTRIBUTING.md criado com sucesso")

def main():
    args = parse_args()
    dry_run = args.dry_run
    
    if dry_run:
        print("MODO DE SIMULAÇÃO: Nenhuma alteração será feita")
    
    # Criar backup se solicitado
    if args.backup:
        backup_path = create_backup(dry_run)
        print(f"Backup criado em: {backup_path}")
    
    # Executar reorganização
    create_directory_structure(dry_run)
    move_files(dry_run)
    remove_files(dry_run)
    update_references(dry_run)
    update_readme(dry_run)
    create_contributing_file(dry_run)
    
    print("\nReorganização concluída!")
    if dry_run:
        print("Este foi apenas um teste, nenhuma alteração real foi feita.")
        print("Execute novamente sem --dry-run para aplicar as alterações.")

if __name__ == "__main__":
    main()
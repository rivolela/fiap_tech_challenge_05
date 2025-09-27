#!/usr/bin/env python3
"""Script para testar se os logs estão sendo salvos na pasta correta"""

import os
import sys
import logging
from dotenv import load_dotenv

# Adicionar o diretório raiz ao PYTHONPATH para importar os módulos do projeto
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging da mesma forma que na API
log_file = os.getenv("LOG_FILE", "logs/api_logs.log")

# Garantir que o diretório de logs existe
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger("test-script")

# Função principal
def main():
    """Função principal para testar o logging"""
    logger.info("Este é um teste para verificar se os logs estão sendo salvos no diretório correto")
    logger.warning("Outro log de teste")
    
    # Verificar se o arquivo de log foi criado
    if os.path.exists(log_file):
        print(f"✅ Arquivo de log criado com sucesso em: {os.path.abspath(log_file)}")
    else:
        print(f"❌ Falha ao criar o arquivo de log em: {os.path.abspath(log_file)}")

if __name__ == "__main__":
    main()
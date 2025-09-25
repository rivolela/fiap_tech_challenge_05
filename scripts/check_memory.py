import os
import sys
import platform
import psutil

def check_memory_usage():
    """Verifica e imprime o uso atual de memória"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"Uso de memória:")
    print(f"  RSS (Resident Set Size): {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"  VMS (Virtual Memory Size): {memory_info.vms / 1024 / 1024:.2f} MB")
    
    # Informações do sistema
    virtual_memory = psutil.virtual_memory()
    print(f"Memória do sistema:")
    print(f"  Total: {virtual_memory.total / 1024 / 1024:.2f} MB")
    print(f"  Disponível: {virtual_memory.available / 1024 / 1024:.2f} MB")
    print(f"  Uso: {virtual_memory.percent}%")

def main():
    """Exibe informações sobre o ambiente de execução da aplicação"""
    print("\n=== INFORMAÇÕES DO AMBIENTE DE EXECUÇÃO ===\n")
    
    # Sistema operacional
    print(f"Sistema operacional: {platform.system()} {platform.release()} {platform.version()}")
    print(f"Python: {platform.python_version()}")
    
    # Diretório atual e variáveis de ambiente
    print(f"\nDiretório de trabalho: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Não definido')}")
    
    # Verificar caminhos críticos
    print("\nVerificando diretórios:")
    critical_paths = [
        "models",
        "data",
        "data/processed",
        "data/metrics",
        "data/logs",
        "data/monitoring"
    ]
    
    for path in critical_paths:
        exists = os.path.exists(path)
        is_dir = os.path.isdir(path) if exists else False
        print(f"  {path}: {'✅ Existe' if exists else '❌ Não existe'} {'(é um diretório)' if is_dir else '(não é um diretório)' if exists else ''}")
        
        # Se não existir e for diretório, criar
        if not exists and path.endswith("/"):
            try:
                os.makedirs(path, exist_ok=True)
                print(f"    ✅ Diretório criado: {path}")
            except Exception as e:
                print(f"    ❌ Erro ao criar diretório: {str(e)}")
    
    # Verificar o modelo
    model_path = "models/scoring_model.pkl"
    model_exists = os.path.exists(model_path)
    print(f"\nModelo: {model_path}: {'✅ Existe' if model_exists else '❌ Não existe'}")
    if model_exists:
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Tamanho: {size_mb:.2f} MB")
    
    # Verificar uso de memória
    print("\n")
    check_memory_usage()
    
    print("\n=== FIM DAS INFORMAÇÕES DE AMBIENTE ===\n")

if __name__ == "__main__":
    main()
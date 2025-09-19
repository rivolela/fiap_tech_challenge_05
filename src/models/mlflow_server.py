"""
mlflow_server.py - Script para iniciar o servidor MLflow

Este script configura e inicia o servidor MLflow para visualização de experimentos.
Também permite excluir experimentos existentes ou configurar o armazenamento remoto.
"""

import os
import subprocess
import sys
import time
import signal
import argparse

def check_mlflow_installed():
    """Verifica se o MLflow está instalado."""
    try:
        import mlflow
        return True
    except ImportError:
        return False

def install_mlflow():
    """Instala o MLflow."""
    print("🔄 Instalando MLflow...")
    subprocess.call([sys.executable, "-m", "pip", "install", "mlflow"])
    print("✅ MLflow instalado com sucesso!")

def start_mlflow_server(port=5000, host='localhost'):
    """Inicia o servidor MLflow."""
    print(f"🚀 Iniciando servidor MLflow em http://{host}:{port}")
    
    # Configurar diretório para armazenar artefatos do MLflow
    os.makedirs('./mlruns', exist_ok=True)
    
    # Definir processo do servidor MLflow
    server_process = subprocess.Popen(
        ["mlflow", "server", "--host", host, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Esperar um pouco para garantir que o servidor iniciou
    time.sleep(2)
    
    if server_process.poll() is None:
        print(f"✅ Servidor MLflow iniciado com sucesso!")
        print(f"🌐 Acesse: http://{host}:{port}")
        print(f"⚠️ Pressione Ctrl+C para encerrar o servidor.")
        
        try:
            # Manter o servidor rodando até o usuário pressionar Ctrl+C
            server_process.wait()
        except KeyboardInterrupt:
            print("🛑 Encerrando servidor MLflow...")
            server_process.terminate()
            server_process.wait()
            print("✅ Servidor MLflow encerrado.")
    else:
        stdout, stderr = server_process.communicate()
        print(f"❌ Erro ao iniciar servidor MLflow:")
        print(stderr)
        return False
    
    return True

def delete_experiment(experiment_name):
    """Exclui um experimento específico."""
    try:
        import mlflow
        
        # Verificar se o experimento existe
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            # Em versões recentes do MLflow, usa-se delete_experiment ao invés de delete
            try:
                mlflow.delete_experiment(experiment.experiment_id)
            except AttributeError:
                # Fallback para versões antigas
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                client.delete_experiment(experiment.experiment_id)
                
            print(f"✅ Experimento '{experiment_name}' excluído com sucesso!")
        else:
            print(f"⚠️ Experimento '{experiment_name}' não encontrado!")
            
    except Exception as e:
        print(f"❌ Erro ao excluir experimento: {str(e)}")

def list_experiments():
    """Lista todos os experimentos existentes."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        experiments = client.list_experiments()
        
        if not experiments:
            print("⚠️ Nenhum experimento encontrado!")
            return
        
        print("\n📊 EXPERIMENTOS EXISTENTES:")
        print("-" * 80)
        print(f"{'ID':<10} {'Nome':<30} {'Artefatos':<40}")
        print("-" * 80)
        
        for exp in experiments:
            artifact_location = exp.artifact_location
            print(f"{exp.experiment_id:<10} {exp.name:<30} {artifact_location:<40}")
            
    except Exception as e:
        print(f"❌ Erro ao listar experimentos: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Gerenciador de MLflow para Decision")
    parser.add_argument("--start", action="store_true", help="Inicia o servidor MLflow")
    parser.add_argument("--port", type=int, default=5000, help="Porta para o servidor MLflow")
    parser.add_argument("--host", type=str, default="localhost", help="Host para o servidor MLflow")
    parser.add_argument("--delete", type=str, help="Nome do experimento a ser excluído")
    parser.add_argument("--list", action="store_true", help="Lista todos os experimentos")
    
    args = parser.parse_args()
    
    # Verificar se MLflow está instalado
    if not check_mlflow_installed():
        print("⚠️ MLflow não está instalado.")
        install_mlflow()
    
    # Executar ação escolhida
    if args.list:
        list_experiments()
    elif args.delete:
        delete_experiment(args.delete)
    elif args.start:
        start_mlflow_server(args.port, args.host)
    else:
        # Por padrão, inicia o servidor
        start_mlflow_server()

if __name__ == "__main__":
    main()
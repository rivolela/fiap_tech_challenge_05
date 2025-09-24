"""
alert_system.py - Sistema de alertas para anomalias e drift do modelo

Este módulo fornece funções para detectar anomalias nas métricas e drift do modelo,
e enviar alertas quando necessário.
"""

import os
import json
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Importar módulos locais
from src.monitoring.metrics_store import get_metrics_history
from src.monitoring.drift_detector import get_latest_drift_report, DRIFT_REPORTS_DIR

# Configurações de alertas
ALERTS_DIR = Path("data/monitoring/alerts")
ALERTS_CONFIG_FILE = ALERTS_DIR / "alert_config.json"
ALERTS_LOG_FILE = ALERTS_DIR / "alert_log.json"

# Inicializar sistema de alertas
def initialize_alert_system():
    """Inicializa o sistema de alertas."""
    os.makedirs(ALERTS_DIR, exist_ok=True)
    
    # Criar arquivo de configuração de alertas se não existir
    if not ALERTS_CONFIG_FILE.exists():
        default_config = {
            "alert_recipients": ["admin@example.com"],
            "thresholds": {
                "drift_score": 0.3,  # Alertar se o score de drift for maior que 0.3
                "accuracy_drop": 0.05,  # Alertar se acurácia cair mais de 5%
                "error_rate": 1.0,  # Alertar se taxa de erro for maior que 1%
                "latency": 200  # Alertar se latência for maior que 200ms
            },
            "smtp_config": {
                "enabled": False,  # Desabilitado por padrão
                "server": "smtp.example.com",
                "port": 587,
                "user": "username",
                "password": "",
                "from_email": "alerts@decision.com"
            }
        }
        
        with open(ALERTS_CONFIG_FILE, 'w') as f:
            json.dump(default_config, f, indent=2)
    
    # Criar log de alertas se não existir
    if not ALERTS_LOG_FILE.exists():
        with open(ALERTS_LOG_FILE, 'w') as f:
            json.dump({
                "alerts": []
            }, f, indent=2)

def get_alert_config() -> Dict[str, Any]:
    """Carrega a configuração de alertas."""
    try:
        with open(ALERTS_CONFIG_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        initialize_alert_system()
        return get_alert_config()

def update_alert_config(new_config: Dict[str, Any]):
    """Atualiza a configuração de alertas."""
    with open(ALERTS_CONFIG_FILE, 'w') as f:
        json.dump(new_config, f, indent=2)

def log_alert(alert_type: str, message: str, details: Dict[str, Any] = None):
    """
    Registra um alerta no log.
    
    Args:
        alert_type: Tipo de alerta (drift, accuracy, error_rate, etc)
        message: Mensagem descritiva do alerta
        details: Detalhes adicionais do alerta
    """
    try:
        with open(ALERTS_LOG_FILE, 'r') as f:
            log_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log_data = {"alerts": []}
    
    alert = {
        "timestamp": datetime.datetime.now().isoformat(),
        "type": alert_type,
        "message": message,
        "details": details or {}
    }
    
    log_data["alerts"].append(alert)
    
    with open(ALERTS_LOG_FILE, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    return alert

def send_email_alert(subject: str, body: str, recipients: List[str]):
    """
    Envia um alerta por email.
    
    Args:
        subject: Assunto do email
        body: Conteúdo do email (pode ser HTML)
        recipients: Lista de destinatários
    
    Returns:
        Tupla com status (bool) e mensagem
    """
    config = get_alert_config()
    smtp_config = config.get("smtp_config", {})
    
    if not smtp_config.get("enabled", False):
        return False, "Sistema de email não está habilitado"
    
    try:
        # Configurar mensagem
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_config.get("from_email", "alerts@decision.com")
        msg["To"] = ", ".join(recipients)
        
        # Adicionar corpo da mensagem
        msg.attach(MIMEText(body, "html"))
        
        # Conectar ao servidor SMTP
        server = smtplib.SMTP(smtp_config.get("server"), smtp_config.get("port", 587))
        server.starttls()
        server.login(smtp_config.get("user"), smtp_config.get("password"))
        
        # Enviar email
        server.sendmail(msg["From"], recipients, msg.as_string())
        server.quit()
        
        return True, "Email enviado com sucesso"
        
    except Exception as e:
        return False, f"Erro ao enviar email: {str(e)}"

def check_drift_alerts() -> List[Dict[str, Any]]:
    """
    Verifica se há drift do modelo que requer alertas.
    
    Returns:
        Lista de alertas gerados
    """
    alerts = []
    config = get_alert_config()
    thresholds = config.get("thresholds", {})
    
    try:
        # Obter o relatório mais recente de drift
        drift_report = get_latest_drift_report()
        
        # Verificar se o score de drift ultrapassou o limiar
        drift_score = drift_report.get("drift_score", 0)
        if drift_score > thresholds.get("drift_score", 0.3):
            # Gerar alerta para drift elevado
            features_with_drift = drift_report.get("features_with_drift", [])
            
            alert = log_alert(
                alert_type="drift",
                message=f"Drift detectado com score {drift_score:.2f}",
                details={
                    "drift_score": drift_score,
                    "features_affected": features_with_drift,
                    "report_timestamp": drift_report.get("timestamp")
                }
            )
            
            # Enviar email se habilitado
            if config.get("smtp_config", {}).get("enabled", False):
                feature_list = "<br>".join([f"- {feature}" for feature in features_with_drift[:5]])
                if len(features_with_drift) > 5:
                    feature_list += f"<br>- ... e outras {len(features_with_drift) - 5} features."
                
                email_body = f"""
                <h2>⚠️ Alerta de Drift do Modelo</h2>
                <p>O sistema detectou um drift significativo no modelo de scoring.</p>
                <p><strong>Score de Drift:</strong> {drift_score:.2f}</p>
                <p><strong>Timestamp:</strong> {drift_report.get("timestamp")}</p>
                <p><strong>Features afetadas:</strong></p>
                {feature_list}
                <br>
                <p>Acesse o dashboard para mais detalhes.</p>
                """
                
                send_email_alert(
                    subject="[ALERTA] Drift Detectado no Modelo de Scoring",
                    body=email_body,
                    recipients=config.get("alert_recipients", [])
                )
            
            alerts.append(alert)
    
    except Exception as e:
        log_alert(
            alert_type="error",
            message=f"Erro ao verificar alertas de drift: {str(e)}"
        )
    
    return alerts

def check_metrics_alerts() -> List[Dict[str, Any]]:
    """
    Verifica se há anomalias nas métricas que requerem alertas.
    
    Returns:
        Lista de alertas gerados
    """
    alerts = []
    config = get_alert_config()
    thresholds = config.get("thresholds", {})
    
    try:
        # Obter histórico de métricas
        metrics_data = get_metrics_history()
        metrics_history = metrics_data.get("metrics_history", [])
        
        if len(metrics_history) < 2:
            return []  # Não há dados suficientes para comparação
        
        # Obter métricas mais recentes e anteriores
        latest_metrics = metrics_history[-1].get("metrics", {})
        previous_metrics = metrics_history[-2].get("metrics", {})
        
        # Verificar queda na acurácia
        latest_accuracy = latest_metrics.get("accuracy", 1.0)
        previous_accuracy = previous_metrics.get("accuracy", 1.0)
        
        accuracy_drop = previous_accuracy - latest_accuracy
        if accuracy_drop > thresholds.get("accuracy_drop", 0.05):
            # Gerar alerta para queda na acurácia
            alert = log_alert(
                alert_type="accuracy",
                message=f"Queda na acurácia detectada: {accuracy_drop:.3f}",
                details={
                    "previous_accuracy": previous_accuracy,
                    "current_accuracy": latest_accuracy,
                    "drop": accuracy_drop,
                    "timestamp": metrics_history[-1].get("timestamp")
                }
            )
            
            # Enviar email se habilitado
            if config.get("smtp_config", {}).get("enabled", False):
                email_body = f"""
                <h2>⚠️ Alerta de Queda na Acurácia</h2>
                <p>O sistema detectou uma queda significativa na acurácia do modelo.</p>
                <p><strong>Acurácia Anterior:</strong> {previous_accuracy:.3f}</p>
                <p><strong>Acurácia Atual:</strong> {latest_accuracy:.3f}</p>
                <p><strong>Queda:</strong> {accuracy_drop:.3f} ({accuracy_drop*100:.1f}%)</p>
                <p><strong>Timestamp:</strong> {metrics_history[-1].get("timestamp")}</p>
                <br>
                <p>Acesse o dashboard para mais detalhes.</p>
                """
                
                send_email_alert(
                    subject="[ALERTA] Queda na Acurácia do Modelo de Scoring",
                    body=email_body,
                    recipients=config.get("alert_recipients", [])
                )
            
            alerts.append(alert)
    
    except Exception as e:
        log_alert(
            alert_type="error",
            message=f"Erro ao verificar alertas de métricas: {str(e)}"
        )
    
    return alerts

def check_all_alerts() -> List[Dict[str, Any]]:
    """
    Verifica todos os tipos de alertas.
    
    Returns:
        Lista combinada de alertas gerados
    """
    drift_alerts = check_drift_alerts()
    metrics_alerts = check_metrics_alerts()
    
    return drift_alerts + metrics_alerts

def get_recent_alerts(days: int = 7) -> List[Dict[str, Any]]:
    """
    Obtém alertas recentes dos últimos dias.
    
    Args:
        days: Número de dias para olhar para trás
        
    Returns:
        Lista de alertas recentes
    """
    try:
        with open(ALERTS_LOG_FILE, 'r') as f:
            log_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    
    alerts = log_data.get("alerts", [])
    
    # Filtrar por data
    cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
    recent_alerts = [
        alert for alert in alerts
        if alert.get("timestamp", "0") >= cutoff_date
    ]
    
    return recent_alerts
# MLflow para Validação de Modelos Decision

Este documento explica como usar o MLflow para validar, testar e monitorar as métricas dos modelos de scoring para a Decision.

## O que é MLflow?

MLflow é uma plataforma open source para gerenciar o ciclo de vida de machine learning, incluindo:

- **Tracking**: Rastreamento de experimentos para registrar e comparar parâmetros e resultados
- **Models**: Gerenciamento e implantação de modelos em diferentes ambientes
- **Projects**: Empacotamento de código ML para facilitar reprodutibilidade
- **Registry**: Armazenamento central para modelos em produção

## Instalação

```bash
pip install mlflow
```

## Utilizando MLflow com o Decision Scoring Model

### 1. Treinamento com Rastreamento

O script `train_simple.py` já está configurado para usar MLflow automaticamente, caso esteja instalado. Para treinar modelos com rastreamento MLflow:

```bash
# Treinar o modelo padrão (RandomForest)
python3 src/models/train_simple.py

# Comparar diferentes modelos (RandomForest e GradientBoosting)
python3 src/models/train_simple.py --compare
```

### 2. Visualizando Experimentos

Para visualizar os experimentos, métricas e artefatos:

```bash
# Iniciar o servidor MLflow
python3 src/models/mlflow_server.py

# Ou diretamente
mlflow ui
```

Após iniciar o servidor, acesse http://localhost:5000 no navegador.

### 3. Gerenciamento de Experimentos

O script `mlflow_server.py` oferece algumas opções para gerenciar experimentos:

```bash
# Listar todos os experimentos
python3 src/models/mlflow_server.py --list

# Excluir um experimento específico
python3 src/models/mlflow_server.py --delete "Decision-Scoring-Model"

# Iniciar servidor em uma porta específica
python3 src/models/mlflow_server.py --port 8888
```

## Métricas Rastreadas

Os seguintes parâmetros e métricas são rastreados durante o treinamento:

### Parâmetros
- Hiperparâmetros do modelo (n_estimators, max_depth, etc.)
- Número de amostras de treinamento
- Número de features categóricas e numéricas

### Métricas
- **AUC-ROC**: Área sob a curva ROC
- **Acurácia**: Percentual de previsões corretas
- **F1-Score**: Média harmônica entre precisão e recall
- **Precisão**: Percentual de verdadeiros positivos entre todos os positivos preditos
- **Recall**: Percentual de verdadeiros positivos capturados
- **Average Precision**: Área sob a curva de precisão-recall

## Artefatos Salvos

Cada experimento salva os seguintes artefatos:

- **Modelo**: O modelo treinado serializado
- **Matriz de Confusão**: Visualização da matriz de confusão
- **Relatório de Classificação**: Métricas detalhadas por classe

## Comparando Modelos

No modo de comparação (`--compare`), o script treina e avalia diferentes algoritmos:

1. **RandomForest**: Bom para lidar com features categóricas e numéricas
2. **GradientBoosting**: Geralmente mais preciso, mas pode ser mais propenso a overfitting

A comparação é baseada na métrica AUC-ROC no conjunto de validação.

## Melhores Práticas

1. **Nomeie Seus Experimentos**: Use nomes descritivos para facilitar a identificação
2. **Compare Modelos**: Use o modo `--compare` para testar diferentes algoritmos
3. **Examine Métricas**: Analise todas as métricas, não apenas a acurácia
4. **Registre Modelos Importantes**: Use o MLflow Model Registry para modelos que irão para produção
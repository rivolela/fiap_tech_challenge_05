# Correção para o MLflow - Métricas Variáveis

## Problema

Foi identificado um problema onde o MLflow sempre registrava as mesmas métricas de desempenho para o modelo, independentemente da quantidade de vezes que o treinamento era executado. Isso ocorria porque o modelo estava sendo criado com uma semente fixa (`random_state=42`), resultando no mesmo comportamento para cada execução.

## Solução Implementada

As seguintes modificações foram feitas no arquivo `src/models/train_simple.py`:

1. **Adição de funções para gerar aleatoriedade:**
   - `generate_random_seed()`: Gera uma semente aleatória com base no timestamp atual
   - `generate_run_name()`: Cria um nome de execução único para o MLflow

2. **Modificação da função de treinamento:**
   - Adicionado parâmetro `random_state` opcional para `train_scoring_model()`
   - Se `random_state=None`, uma semente aleatória é gerada automaticamente
   - Hiperparâmetros do modelo agora variam com base na semente aleatória

3. **Nomes de runs únicos no MLflow:**
   - Cada execução agora tem um nome único baseado no tipo de modelo, timestamp e ID aleatório
   - O `random_state` usado é registrado como parâmetro para rastreabilidade

4. **Correções para problemas de tipos:**
   - Todas as métricas são convertidas explicitamente para `float` para evitar erros de tipo ao registrar no MLflow
   - Adicionado tratamento de erro para lidar com diferentes versões do MLflow

5. **Interface de linha de comando aprimorada:**
   - Adicionado argumento `--random-seed` opcional para permitir execuções reproduzíveis quando necessário

## Como Testar

Foi criado um script de teste `scripts/test_random_seed.sh` que executa o treinamento várias vezes:
- 3 vezes com sementes aleatórias (cada execução deve ter métricas diferentes)
- 2 vezes com a mesma semente fixa (ambas execuções devem ter métricas idênticas)
- 1 vez com uma semente fixa diferente

### Para testar:

1. Execute o script de teste:
   ```bash
   ./scripts/test_random_seed.sh
   ```

2. Visualize os resultados no MLflow UI:
   ```bash
   mlflow ui --port 5001
   ```
   
3. Navegue até http://localhost:5001 e verifique o experimento "Decision-Scoring-Model"

### Resultados Esperados

- As execuções 1, 2 e 3 devem ter métricas diferentes entre si
- As execuções 4 e 5 devem ter métricas idênticas (mesmo seed=42)
- A execução 6 deve ter métricas diferentes das outras (seed=123)

## Benefícios

- **Maior confiabilidade**: Cada treinamento agora gera resultados ligeiramente diferentes, permitindo uma melhor avaliação da robustez do modelo
- **Experimentos significativos**: O MLflow agora pode mostrar a variação real entre diferentes execuções do modelo
- **Flexibilidade**: Ainda é possível usar uma semente fixa quando a reprodutibilidade é necessária
- **Visualização melhorada**: Cada execução tem um nome único e timestamp para melhor organização no MLflow UI
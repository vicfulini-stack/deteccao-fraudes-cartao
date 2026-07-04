# Detecção de Anomalias em Transações Financeiras com Machine Learning

Este projeto foi desenvolvido como parte dos requisitos do bootcamp para a identificação de fraudes em transações de cartão de crédito. O objetivo principal é utilizar técnicas avançadas de Machine Learning para lidar com um cenário de dados altamente desbalanceados e garantir uma alta taxa de detecção de fraudes.

##  Estrutura do Projeto

*   `main.py`: Script principal contendo o pipeline de carregamento, tratamento de dados, treinamento do modelo e avaliação de métricas.
*   `.gitignore`: Configuração para garantir a segurança do repositório, impedindo o envio de arquivos temporários, caches e dados locais sensíveis.

##  O Problema dos Dados Desbalanceados

Ao analisar o conjunto de dados fornecido pelo TensorFlow (`creditcard.csv`), foi identificado um desbalanceamento extremo entre as classes:
*   **Transações Legítimas (Classe 0):** 284.315
*   **Transações Fraudulentas (Classe 1):** 492

Em cenários como este, métricas tradicionais como a *Acurácia* são enganosas. Por isso, o projeto focou em otimizar as métricas de **Precision (Precisão)** e, principalmente, **Recall (Sensibilidade)**, garantindo que o maior número possível de fraudes seja interceptado.

##  Tecnologias e Bibliotecas Utilizadas

*   **Python 3**
*   **Pandas & NumPy**: Manipulação e análise exploratória dos dados.
*   **Scikit-Learn**: Pré-processamento (`StandardScaler`), divisão estratificada dos dados (`train_test_split`) e modelagem preditiva.
*   **Seaborn & Matplotlib**: Visualização de dados e geração de gráficos de desempenho.

##  Evolução do Modelo e Resultados

Durante o desenvolvimento do projeto, foram exploradas diferentes abordagens de Machine Learning para encontrar o melhor equilíbrio de negócio:

### 1. Abordagem Inicial: Isolation Forest (Não Supervisionado)
Inicialmente, foi aplicado o algoritmo *Isolation Forest* com uma taxa de contaminação estimada. O modelo serviu como uma excelente linha de base, mas por não utilizar os rótulos históricos das fraudes, apresentou oportunidades de melhoria:
*   **Recall:** 0.34 (identificava apenas 34% das fraudes existentes).

### 2. Abordagem Final: RandomForestClassifier (Supervisionado + Balanceado)
Para maximizar a detecção de fraudes, alteramos a abordagem para o modelo supervisionado *Random Forest*, utilizando o parâmetro de compensação nativa `class_weight='balanced'`. Isso penaliza severamente os erros na classe minoritária, resultando em um ganho massivo de performance:
*   **Precision:** 0.91 (91% das transações classificadas como fraude eram realmente fraudes).
*   **Recall:** 0.79 (o modelo conseguiu capturar **79% de todas as fraudes** do conjunto de teste).

---
*Nota: A Matriz de Confusão gerada pelo script demonstra visualmente a redução drástica de falsos negativos (apenas 21 fraudes não detectadas), consolidando um modelo altamente eficiente para implementação em cenários reais de análise de risco.*

﻿# Projeto de Classificação com Aprendizado de Máquina

Este projeto realiza o processamento, balanceamento e classificação de dados de tráfego de rede utilizando técnicas de aprendizado de máquina. O objetivo é comparar diferentes algoritmos de classificação e avaliar seu desempenho em um conjunto de dados multiclasse.

## Estrutura do Projeto

- **script.py**: Script principal para treinamento, validação e comparação dos modelos.
- **script_jpt.ipynb**: Notebook com experimentos, análises e pré-processamento dos dados.
- **subamostragem/**: Contém arquivos CSV e Parquet com dados balanceados e descartados.
- **models/**: Pasta destinada ao salvamento dos modelos treinados.
- **requirements.txt**: Lista de dependências do projeto.
- **Dockerfile**: Permite a execução do projeto em ambiente Docker para maior reprodutibilidade.

## Principais Funcionalidades

- Leitura e limpeza de dados (remoção de NaN e infinitos).
- Balanceamento das classes utilizando técnicas como NearMiss e ClusterCentroids.
- Redução de dimensionalidade e otimização dos tipos de dados para economia de memória.
- Treinamento e avaliação de múltiplos modelos: KNN, Decision Tree, Random Forest, XGBoost e AutoSklearn.
- Validação cruzada (KFold) e comparação estatística dos resultados (teste de Wilcoxon).
- Salvamento dos melhores modelos e métricas em arquivos CSV.

## Como Executar

1. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```
2. (Opcional) Construa e execute via Docker:
   ```
   docker build -t autosklearn-image .
   docker run -it -v "$(pwd):/app" autosklearn-image
   ```
3. Execute o script principal:
   ```
   python script.py
   ```

## Resultados

- As métricas de desempenho dos modelos são salvas em `metrics.csv`.
- Resultados estatísticos dos testes de comparação estão em `wilcoxon_results.csv`.
- Os melhores modelos de cada algoritmo são salvos na pasta raiz do projeto.

## Observações

- Os arquivos de dados `.parquet` e `.csv` são gerados e utilizados durante o processamento.

---

# Visão Geral do Projeto de Predição de Preço de Casas

Este projeto tem como objetivo a construção de um modelo de aprendizado de máquina para prever o preço de venda de casas com base em diversas características. Os dados utilizados neste projeto são provenientes de uma competição de predição de preço de casas, onde são fornecidos dados de treino e teste, contendo informações sobre propriedades como o número de quartos, área total do terreno, tipo de acabamento, entre outras.

## Etapas do Projeto

### 1. **Carregamento dos Dados**
O primeiro passo no projeto foi carregar os dados de treinamento (`train.csv`) e de teste (`test.csv`) de um diretório de entrada. Ambos os conjuntos de dados foram lidos e combinados em um único DataFrame para facilitar o tratamento conjunto de valores ausentes.

```python
df_1 = pd.read_csv("/kaggle/input/house-price-prediction/train.csv")
df_2 = pd.read_csv("/kaggle/input/house-price-prediction/test (1).csv")
```

### 2. Tratamento de Dados Faltantes
Os dados de entrada apresentam algumas lacunas que precisaram ser tratadas:

- Foi verificado a presença de valores ausentes em diversas colunas.
- Para as colunas categóricas, valores ausentes foram preenchidos com o valor 'null'.
- As colunas com mais de 1100 valores ausentes foram descartadas.
- Valores ausentes em variáveis numéricas foram preenchidos com a média ou moda dos valores da coluna, conforme o caso.

```python
df_objects = df[df.select_dtypes(include=['object']).columns]
df_objects = df_objects.fillna('null')
```

### 3. Transformação de Variáveis Categóricas
As colunas categóricas foram transformadas em variáveis binárias (usando a técnica de codificação "one-hot"), o que possibilitou a utilização dessas variáveis em modelos de aprendizado de máquina.

```python
df_objects_encoded = pd.get_dummies(df_objects)
```

### 4. Divisão dos Dados em Conjunto de Treinamento e Teste
Após o pré-processamento dos dados, o conjunto foi dividido em dois: treinamento e teste, com 80% dos dados sendo usados para treinamento e 20% para teste.

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
```

### 5. Treinamento e Avaliação de Modelos
Foram utilizados três modelos diferentes para prever o preço das casas:

Regressão Linear: Um modelo simples de regressão linear.
XGBoost: Um modelo avançado baseado em árvores de decisão, conhecido por sua alta performance em tarefas de predição.
Random Forest: Um ensemble de árvores de decisão, utilizado para melhorar a precisão e reduzir o overfitting.

Os modelos foram avaliados utilizando o erro quadrático médio (MSE).
```python
model_1 = LinearRegression()
model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.1, random_state=42)
model_3 = RandomForestRegressor(n_estimators=1000)
```

### 6. Salvar o Modelo Treinado
O modelo XGBoost foi selecionado como o melhor e foi salvo em um arquivo .pkl para utilização futura.
```python
import pickle
with open(model_filename, 'wb') as model_file:
    pickle.dump(model_2, model_file)
```

### 7. Previsão e Geração do Arquivo de Saída
Após treinar o modelo, o modelo foi utilizado para fazer previsões nos dados de teste. As previsões, junto com o identificador de cada propriedade, foram salvas em um arquivo CSV (output.csv), pronto para ser submetido ou utilizado.
```python
final.to_csv('output.csv', index=False)
```

### 8. Visualização dos Resultados
Por fim, uma visualização gráfica foi criada para comparar os preços originais das casas com os valores previstos pelo modelo XGBoost.
```python
plt.plot(np.arange(len(Y_test)), Y_test, label='Original')
plt.plot(np.arange(len(Y_test)), y_pred, label='Predicted')
```

Conclusão
Neste projeto, foi desenvolvido um pipeline completo de pré-processamento, treinamento e avaliação de modelos para prever o preço de casas. A combinação de diferentes modelos de aprendizado de máquina (como Regressão Linear, XGBoost e Random Forest) demonstrou a viabilidade de melhorar a precisão da predição de preços imobiliários.

O modelo treinado foi exportado e está disponível para uso posterior.

Tecnologias Utilizadas
Python (para o processamento de dados e treinamento de modelos)
pandas (para manipulação de dados)
seaborn e matplotlib (para visualização dos dados)
scikit-learn (para modelos de aprendizado de máquina)
XGBoost (modelo avançado de árvores de decisão)
pickle (para salvar o modelo treinado)

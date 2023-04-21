# Previsão de inundações com dados de pluviosidade e nível do rio Pavia
# Passo 1: Importação das bibliotecas necessárias

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  

# Passo 2: Carregar e visualizar os dados de precipitação
# Leitura dos Dados de Pluviosidade 
df_pluviosidade = pd.read_csv("Pavia-Quantidade_chuva_por_hora.csv")
print(df_pluviosidade.head())
print(df_pluviosidade.shape)
print(df_pluviosidade.describe())

df_pluviosidade.plot(x='Data/Hora', y='Precipitação acumulada (mm)', style='o')  

plt.title('Precipitação')  
plt.xlabel('Data')  
plt.ylabel('Precipitação em mm')  
plt.show()  


# ## Passo 3: Carregar e visualizar os dados do nível do rio
# Leitura dos Dados do Rio
df_rio = pd.read_csv("Pavia-Nivel_do_rio_por_hora.csv")

print(df_rio.head())
print(df_rio.shape)
print(df_rio.describe())

df_rio.plot(x='Data/Hora', y='Nível (m)', style='o')  
plt.title('Nível do rio')  
plt.xlabel('Data')  
plt.ylabel('Nível Máximo')  
plt.show()  


# Passo 4: Combinar os dois conjuntos de dados 
#df_rio["Data/Hora"] = df_rio["Data/Hora"].str.replace("00:00", "")
df = pd.merge(df_pluviosidade, df_rio, how='outer', on=['Data/Hora'])
print(df.head())
df.plot(x='Precipitação acumulada (mm)', y='Nível (m)', style='o')  
plt.title('Nível do rio')  
plt.xlabel('Precipitação')  
plt.ylabel('Nível Máximo')  
plt.show()  

# Passo 5: Limpar e preparar os dados
df['Precipitação acumulada (mm)'] = df['Precipitação acumulada (mm)'].fillna(0)
df['Nível (m)'] = df['Nível (m)'].fillna(0)
print(df.head())
#df = df[(df != 0).all(1)]
df = df.drop(columns=['Precipitação atual (mm)', 'Data/Hora'])
print(df.shape)
X = df.iloc[:, :1].values
y = df.iloc[:, 1:2].values
#print(X)
#print(y)

# Passo 6: Dividir os dados em conjuntos de treino e teste 
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

# Passo 7: Treinar o modelo de regressão linear 
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train)  
print(regressor.intercept_)
print(regressor.coef_)  
y_pred = regressor.predict(X_test) 

# Passo 8: Realizar previsões e mostrar os resultados
plt.scatter(X_train, y_train)
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Precipitação Vs Nível Rio (Training set)')
plt.xlabel('Precipitação')
plt.ylabel('Nível Rio')
plt.show()
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Precipitação Vs Nível Rio (Test set)')
plt.xlabel('Precipitação')
plt.ylabel('Nível Rio')
plt.show()

# Passo 9: Testar o modelo com novos dados de precipitação 
#@title Insira a quantidade de precipitação em mm
Quantidade_de_Precipitação =  [[22]]
nível_do_rio_previsto = regressor.predict(Quantidade_de_Precipitação)
print(nível_do_rio_previsto)

# Passo 10: Avaliar a possibilidade de inundação 
if (nível_do_rio_previsto > 1.5):
  print("INUNDAÇÃO!!")
else:
  print("Não há inundação")






import pandas as pd 
from sklearn.model_selection import RandomizedSearchCV

dataframe = pd.read_csv('/home/yuri/backup do windows/backup do windows/backup/dados_para_ia/módulo-1/regressão_logistica/Data_train_reduced.csv')

dataset = dataframe.drop(['Product'], axis= 1)

faltantes = dataframe.isnull().sum()
percentual = 100*faltantes/len(dataframe['Product'])
print(percentual)

newdataframe = dataframe.drop(['q8.2','q8.8','q8.9','q8.10', 'q8.17', 'q8.18', 'q8.20', 'Respondent.ID','Product', 'q1_1.personal.opinion.of.this.Deodorant'], axis = 1)
newdataframe['q8.7'] = newdataframe['q8.7'].fillna(newdataframe['q8.7'].median())
newdataframe['q8.12'] = newdataframe['q8.12'].fillna(newdataframe['q8.12'].median())

porcentual = newdataframe.isnull().sum()*100/len(newdataframe['Product.ID'])

y = newdataframe['Instant.Liking'] #target 
x = newdataframe.drop('Instant.Liking', axis = 1)

from sklearn.preprocessing import MinMaxScaler 
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 

normalizador = MinMaxScaler(feature_range = (0,1))
x_norm = normalizador.fit_transform(x)

valores_k = np.array([3, 5, 7, 9, 11])
valores_p = ['minkowski', 'chebyshev']
distancias = np.array([1, 2, 3, 4])

parametros = {'n_neighbors':valores_k, 'p': distancias, 'metric':valores_p}

modelo = KNeighborsClassifier()

random= RandomizedSearchCV(estimator = modelo, 
param_distributions=parametros, 
n_iter= 50, 
cv = 5)

random.fit(x_norm, y)

print(f"Melhores parâmetros: {random.best_params_}")
print(f"Melhor acurácia: {random.best_score_}")

#link do dataset: https://www.kaggle.com/datasets/ramkumarr02/deodorant-instant-liking-data
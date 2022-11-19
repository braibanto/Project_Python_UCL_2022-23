import datetime as dt
import math
import numpy as np
import pandas as pd
import pandas_datareader as pdr

start = dt.datetime(2020, 1, 1)
stop = dt.datetime(2021, 1, 1)

# Initialisation des DataFrame result et data_df
# result[] va contenir le DataFrame finale avec les prix journalier des actions
# data_df est un DataFrame temporaire qui va être alimenté par la methode pdr.get_data_yahoo
# seulement data_df["Adj Close"] nous intéresse pour construire le DataFrame result[]
# la liste stocks[] contient les tickers des actions du portefeuille

result = pd.DataFrame()
data_df = pd.DataFrame()
stocks = ["MSFT", "IBM", "AAPL", "TSLA"]

      # "META", "GOOG", "AMZN", "JPM", "V",
          #"WMT"]

# Construction du DataFrame result[]:
for stock in stocks:

      data_df = pdr.get_data_yahoo (stock, start, stop)
      result[stock] = data_df["Adj Close"]
      data_df = pd.DataFrame()

# Voici un overview du DataFrame result[]:
result.info()

# Nous transformons le DataFrame result[] en un Numpy Array (arr) pour faciliter les calculs
arr = result.to_numpy()

print("shape of arr: ", arr.shape)
print("nombre d'axe (colonne): ", arr.ndim)
print("Nombre d'éléments dans arr: ", arr.size)

moy_arr = np.mean(arr, axis=0)
var_arr = np.var(arr, axis=0)
covar = np.cov(arr.T)

# Calcul des log returns dans un np.array (log_return)
# le return du premier jour est fixé à 0

log_return = np.zeros([252,len(stocks)])
log_return[0,0:] = 0
for j in range(0, len(stocks)):
      for i in range(1, 252):
            log_return[i,j] = math.log(arr[i, j]/arr[i-1, j])
            print(f"log_return [{i},{j}]", log_return[i, j])

# Calcul des returns simples sur base des log returns:
# formule: ret_simple = exp(log_return) - 1

ret_simple = np.zeros([252, len(stocks)])
ret_simple = np.exp(log_return) - 1

# Génération de la matrice colonne contenant les poids des actions dans le portefeuille
weights = np.random.uniform(low=0.01, high=1, size=len(stocks))
sum_weights = sum(weights)
weights_std = weights / sum(weights)

print("Pondérations non-normalisées: ", weights)
print("Somme des pondérations non-normalisées: ", sum_weights)
print("Pondérations normalisées: ", weights_std)
print("Somme des pondérations normalisées: ", sum(weights_std))







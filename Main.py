import datetime as dt
import math
import scipy.optimize as scy
import numpy as np
import pandas as pd
import pandas_datareader as pdr


def fetch_stock_price(stocks, start_date, end_date):

    """ crée un dataframe contenant les prix des actions (adjusted close price)
      INPUT:
            - stock : list (of strings)
            - start_date : date de début (utilise le module datetime as dt)
            - end_date : date de fin (utilise le module datetime as dt)
      OUTPUT:
            - stock_info : dataframe contenant les informations sur les actions """

    stock_price = pdr.get_data_yahoo(stocks, start=start_date, end=end_date)
    stock_price = stock_price["Adj Close"]
    return stock_price


def fetch_stock_returns(stocks, start_date, end_date):

    """crée un dataframe contenant la moyenne des daily returns des actions
    INPUT:
          - stock : list (of strings)
          - start_date : date de début (utilise le module datetime as dt)
          - end_date : date de fin (utilise le module datetime as dt)
    OUTPUT:
          - moy_return : dataframe contenant les moyennes des returns journaliers
          - matrix_cov : dataframe contenant la matrice des variances/covariances des returns journaliers"""

    stock_price = pdr.get_data_yahoo(stocks, start=start_date, end=end_date)
    stock_price = stock_price["Adj Close"]
    stock_returns = stock_price.pct_change()
    moy_return = stock_returns.mean()
    matrix_cov = stock_returns.cov()
    return moy_return, matrix_cov


def ponderation_random_stardardized(stocks):
    """créer un np.array de pondérations aléatoires standardisées (somme = 100%)
    On utilise à cette fin une distribution aléatoire uniforme [0,1[
    INPUT:
        - stocks: liste des actions à pondérer
    OUTPUT:
        - poids_std: np.array contenant les pondérations aléatoires standardiséés """

    poids = np.random.uniform(low=0.01, high=1, size=len(stocks))
    poids_std = poids / sum(poids)
    return poids_std


def perf_porfolio(poids, moy_return, cov_matrix):

    """ Calcul la performance et le risque du portefeuille en tenant de pondération individuelle des action
     INPUT:
        - poids: dataframe contenant les pondérations du portefeuille
        - moy_return: datafram contenant les moyennes journalières des actions
        - cov_matrix: dataframe contenant la matrice var/cov des returns journaliers
    OUTPUT:
        - return_portfolio: un float égale au return annuel pondéré du portefeuille
        - risk_portfolio: un float égale au risque du return annuel pondéré du portefeuille """

    return_portfolio = np.sum(moy_return * poids) * 252
    risk_porfolio = np.sqrt(np.dot(poids.T, np.dot(cov_matrix, poids))) * np.sqrt(252)
    return return_portfolio, risk_porfolio

def sharp_ratio_opp(poids, stock_return, cov_matrix, taux_ss_risque = 0):
    """ Calcul du Sharp ratio (négatif! - voir scipy) d'un portefeuille pour une pondération donnée
    INPUT:
        - poids: np.array contenant la pondération des actions
        - stock_return: array contenant la moyenne """

    port_ret, risk_port = perf_porfolio(poids, stock_return, cov_matrix)
    sharp_ratio = (perf_ret - taux_ss_risque) / risk_port
    return - sharp_ratio



stock_list = ["MSFT", "IBM", "AAPL", "TSLA", "META", "GOOG", "AMZN", "JPM", "V",
              "WMT", "BABA", "CMCSA"]
start_d = dt.datetime(2020, 1, 1)
stop_d = dt.datetime(2021, 1, 31)

ret_stocks, cov_mat = fetch_stock_returns(stock_list, start_d, stop_d)
poids_port = ponderation_random_stardardized(stock_list)
ret_port, risk_port = perf_porfolio(poids_port, ret_stocks, cov_mat)

print(poids_port)
print("Check poids = ", sum(poids_port)*100, " %")
print("Return_porfolio: ", ret_port * 100)
print("Risk portfolio: ", risk_port * 100)



# print(fetch_data(stock_list, start_d, stop_d))

# for stock in stocks:
#
#       data_df = pdr.get_data_yahoo (stock, start, stop)
#       result[stock] = data_df["Adj Close"]
#       data_df = pd.DataFrame()
#
# # Voici un overview du DataFrame result[]:
# result.info()
#
# # Nous transformons le DataFrame result[] en un Numpy Array (arr) pour faciliter les calculs
# arr = result.to_numpy()
#
# print("shape of arr: ", arr.shape)
# print("nombre d'axe (colonne): ", arr.ndim)
# print("Nombre d'éléments dans arr: ", arr.size)
#
# moy_arr = np.mean(arr, axis=0)
# var_arr = np.var(arr, axis=0)
# covar = np.cov(arr.T)
#
# # Calcul des log returns dans un np.array (log_return)
# # le return du premier jour est fixé à 0
#
# log_return = np.zeros([252,len(stocks)])
# log_return[0,0:] = 0
# for j in range(0, len(stocks)):
#       for i in range(1, 252):
#             log_return[i,j] = math.log(arr[i, j]/arr[i-1, j])
#             print(f"log_return [{i},{j}]", log_return[i, j])
#
# # Calcul des returns simples sur base des log returns:
# # formule: ret_simple = exp(log_return) - 1
#
# ret_simple = np.zeros([252, len(stocks)])
# ret_simple = np.exp(log_return) - 1
#
# # Génération de la matrice colonne contenant les poids des actions dans le portefeuille
# weights = np.random.uniform(low=0.01, high=1, size=len(stocks))
# sum_weights = sum(weights)
# weights_std = weights / sum(weights)
#
#
# print("Pondérations non-normalisées: ", weights)
# print("Somme des pondérations non-normalisées: ", sum_weights)
# print("Pondérations normalisées: ", weights_std)
# print("Somme des pondérations normalisées: ", sum(weights_std))
#
#
# # Calcul du rendement pondéré du portefeuille (methode np.average)
# rdt_portefeuille = np.zeros([252,1])
# rdt_portefeuille = np.average(ret_simple, axis=1, weights=weights_std)
#
# # Calcul de la rentabilité annuelle moyenne du portefeuille
# moy_rdt_annuel_portefeuille = np.mean(rdt_portefeuille, axis=0) * 252
# print("Rendement moyen pondéré du portefeuille: ", moy_rdt_annuel_portefeuille)
#
# # Calcul de l'écart type du rdt du portefeuille
# std_rdt_annuel_portefeuille = np.std(rdt_portefeuille, axis=0) * 252
# print("Risque du portefeuille: ", std_rdt_annuel_portefeuille)
#
# # Calcul du ratio de Sharp
# sr_portefeuille = moy_rdt_annuel_portefeuille / std_rdt_annuel_portefeuille
# print("Ration de sharpe du portefeuille: ",sr_portefeuille)

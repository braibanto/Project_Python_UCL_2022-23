import datetime as dt
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
    stock_log_returns = np.log(stock_price / stock_price.shift(1)).dropna()
    moy_log_return = stock_log_returns.mean()
    matrix_cov = stock_log_returns.cov()
    return moy_log_return, matrix_cov


def ponderation_random_stardardised(stocks):

    """ Crée un np.array de pondérations aléatoires standardisées (somme = 100%)
    On utilise à cette fin un générateur de nombre aléatoire [0,1]
    INPUT:
        - stocks: liste des actions à pondérer
    OUTPUT:
        - poids: np.array contenant les pondérations aléatoires standardisées """

    poids = np.random.random(len(stocks))
    poids /= np.sum(poids)
    return poids


def perf_porfolio(poids, moy_return, cov_matrix):

    """ Calcul la performance et le risque du portefeuille en tenant de pondération individuelle des action
     INPUT:
        - poids: dataframe contenant les pondérations du portefeuille
        - moy_return: dataframe contenant les moyennes journalières des actions
        - cov_matrix: dataframe contenant la matrice var/cov des returns journaliers
    OUTPUT:
        - return_portfolio: un float égale au return annuel pondéré du portefeuille
        - risk_portfolio: un float égale au risque du return annuel pondéré du portefeuille """

    return_portfolio = np.sum(moy_return * poids) * 252
    risk_porfolio = np.sqrt(np.dot(poids.T, np.dot(cov_matrix, poids))) * np.sqrt(252)
    return return_portfolio, risk_porfolio


def sharp_ratio_opp(poids, stock_return, cov_matrix, taux_ss_risque=0):
    """ Calcul du Sharp ratio (négatif ! - voir scipy) d'un portefeuille pour une pondération donnée
    INPUT:
        - poids: np.array contenant la pondération des actions
        - stock_return: array contenant la moyenne """

    port_ret, risk_port = perf_porfolio(poids, stock_return, cov_matrix)
    sharp_ratio = (port_ret - taux_ss_risque) / risk_port
    return - sharp_ratio

# MAIN-------------------------------------------------------------------------


stock_list = ["MSFT", "IBM", "AAPL", "TSLA", "META", "GOOG", "AMZN", "JPM", "V",
              "WMT", "BABA", "CMCSA"]
start_d = dt.datetime(2020, 1, 1)
stop_d = dt.datetime(2021, 1, 31)

ret_stocks, cov_mat = fetch_stock_returns(stock_list, start_d, stop_d)
poids_port = ponderation_random_stardardised(stock_list)
return_portefeuille, risk_portefeuille = perf_porfolio(poids_port, ret_stocks, cov_mat)

print(poids_port)
print("Check poids = ", sum(poids_port)*100, " %")
print("Return portfolio: ", return_portefeuille * 100)
print("Risk portfolio: ", risk_portefeuille * 100)

import datetime as dt

import scipy.optimize
import scipy.optimize as scy
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf

def fetch_stock_price(stocks, start_date, end_date):

    """ crée un dataframe contenant les prix des actions (adjusted close price)
      INPUT:
            - stock : list (of strings)
            - start_date : date de début (utilise le module datetime as dt)
            - end_date : date de fin (utilise le module datetime as dt)
      OUTPUT:
            - stock_info : dataframe contenant les informations sur les actions """

    stock_price = pdr.DataReader (stocks, data_source='stooq', start=start_date, end=end_date)
    stock_price = stock_price[::-1]
    stock_price = stock_price['Close']
    return stock_price


def calc_stock_returns(stock_list, start_date, end_date):

    """crée un dataframe contenant la moyenne des daily log_returns des actions
    INPUT:
          - stock : ticker des actions dans le scope de cette analyse (list of strings)
          - start_date : date de début (utilise le module datetime as dt)
          - end_date : date de fin (utilise le module datetime as dt)
    OUTPUT:
          - moy_return : dataframe contenant les moyennes des log_returns journaliers
          - matrix_cov : dataframe contenant la matrice des variances/covariances des log_returns journaliers"""

    stock_price = pdr.DataReader (stock_list, data_source='stooq', start=start_date, end=end_date)
    stock_price = stock_price[::-1]
    stock_price = stock_price['Close']

    log_returns = np.log(stock_price / stock_price.shift(1))
    moy_log_return = log_returns.mean()
    matrix_cov = log_returns.cov()

    return moy_log_return, matrix_cov
#
#
def poids_random_stardardised(stock_list):

    """ Crée un np.array de pondérations aléatoires standardisées (somme = 100%)
    On utilise à cette fin un générateur de nombre aléatoire [0,1]
    INPUT:
        - stocks: liste des actions à pondérer
    OUTPUT:
        - poids: np.array contenant les pondérations aléatoires standardisées """

    poids = np.random.random(len(stock_list))
    poids /= np.sum(poids)
    return poids


def perf_portfolio(poids, moy_return, cov_matrix):

    """ Calcul la performance annuelle et le risque du portefeuille en tenant compte de la pondération individuelle des actions
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

def variance_port(poids, moy_return, cov_matrix):
    return perf_portfolio(poids, moy_return, cov_matrix)[1]


def sharp_ratio_opp(poids, moy_return, cov_matrix, ss_risque=0):
    """ Calcul du Sharp ratio (négatif ! - voir scipy) d'un portefeuille pour une pondération donnée
    INPUT:
        - poids: np.array contenant la pondération des actions
        - stock_return: array contenant la moyenne """

    p_ret, r_port = perf_portfolio(poids, moy_return, cov_matrix)
    sharp_ratio = - (p_ret - ss_risque) / r_port
    return sharp_ratio

def max_sharp_ratio(moy_return, cov_matrix, ss_risque=0, constraints_set=(0,1)):

    long_port = len(moy_return)
    args = (moy_return, cov_matrix, ss_risque)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraints_set
    bounds = tuple(bound for asset in range(long_port))
    result = scy.minimize(sharp_ratio_opp, long_port*[1./long_port], args=args, method='SLSQP',
                                     bounds=bounds, constraints=constraints)

    poids_result_df = pd.DataFrame(result['x'], index=moy_return.index, columns=['poids'])
    print(round(poids_result_df*100,2))
    return result, poids_result_df

def minimum_variance(moy_return, cov_matrix, constraints_set=(0,1)):
    long_port = len(moy_return)
    init_guess = np.random.random (long_port)
    init_guess /= init_guess.sum ()
    args = (moy_return, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraints_set
    bounds = tuple(bound for asset in range(long_port))
    result = scy.minimize(variance_port, long_port * [1./long_port], args=args, method='SLSQP',
                           bounds=bounds, constraints=constraints)

    result_df = pd.DataFrame (result['x'], index=moy_return.index, columns=['poids'])
    print(round(result_df*100,2))
    return result

# MAIN-------------------------------------------------------------------------
stock_list = ["MSFT", "IBM", "META", "V"]
start_d = dt.datetime(2017, 1, 1)
stop_d = dt.datetime(2017, 12, 31)

price_matrix = fetch_stock_price(stock_list, start_d, stop_d)
result_calc = calc_stock_returns(stock_list, start_d, stop_d)
# print(price_matrix)
#print(round(result_calc[1]*252*100,2))
#print(round(result_calc[0]*252*100, 2))


#weights = poids_random_stardardised(stock_list)
#port_perf = perf_portfolio(weights, result_calc[0], result_calc[1])
#print("poids: : ", weights)
#print("type port_perf: ", port_perf)
#print(type(port_perf))

optimi_SR = max_sharp_ratio(result_calc[0], result_calc[1])
print(type(optimi_SR))
print("optimisation SR :", optimi_SR[0])

print("Poids Optimisation SR:")
print("optimisation SR - poids: ", optimi_SR[1])
print("Perf Portfolio Optimisation SR", np.dot(result_calc[0],optimi_SR[1]))


minimi_var = minimum_variance(result_calc[0], result_calc[1])
print(type(minimi_var))
print("minimiation variance: ",minimi_var)



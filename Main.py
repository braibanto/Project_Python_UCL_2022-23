import datetime as dt
import scipy.optimize as scy
import numpy as np
import pandas as pd
#import pandas_datareader as pdr
import yfinance as yf
import matplotlib.pyplot as plt


def fetch_stock_price(stocks, start_date, end_date):

    """ crée un dataframe contenant les prix des actions (adjusted close price)
      INPUT:
            - stock : list (of strings)
            - start_date : date de début (utilise le module datetime as dt)
            - end_date : date de fin (utilise le module datetime as dt)
      OUTPUT:
            - stock_info : dataframe contenant les informations sur les actions """

    stock_price = yf.download(stocks, start=start_date, end=end_date, progress=True)
    stock_price = stock_price['Close']
    return stock_price

def calc_stock_returns(stocks, start_date, end_date):

    """crée un dataframe contenant la moyenne des daily log_returns des actions
    INPUT:
          - stock : ticker des actions dans le scope de cette analyse (list of strings)
          - start_date : date de début (utilise le module datetime as dt)
          - end_date : date de fin (utilise le module datetime as dt)
    OUTPUT:
          - moy_return : dataframe contenant les moyennes des log_returns journaliers
          - matrix_cov : dataframe contenant la matrice des variances/covariances des log_returns journaliers"""

    stock_price = yf.download(stocks, start=start_date, end=end_date, progress=True)
    stock_price = stock_price['Close']

    log_returns = np.log(stock_price / stock_price.shift(1))
    moy_log_return = log_returns.mean()
    matrix_cov = log_returns.cov()

    return moy_log_return, matrix_cov

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
    risk_porfolio = np.dot(np.dot(cov_matrix, poids),poids)**(1/2) * np.sqrt(252)
    return return_portfolio, risk_porfolio
#
def poids_random(stock_list):

    """ Crée un np.array de pondérations aléatoires standardisées (somme = 100%)
    On utilise à cette fin un générateur de nombre aléatoire [0,1]
    INPUT:
        - stocks: liste des actions à pondérer
    OUTPUT:
        - poids: np.array contenant les pondérations aléatoires standardisées """

    poids = np.random.random(len(stock_list))
    poids /= np.sum(poids)
    return poids

# def poids_random_short(stock_list):
#
#     """ Crée un np.array de pondérations aléatoires standardisées (somme = 100%)
#     On utilise à cette fin un générateur de nombre aléatoire [0,1]
#     INPUT:
#         - stocks: liste des actions à pondérer
#     OUTPUT:
#         - poids: np.array contenant les pondérations aléatoires standardisées """
#
#     poids = np.random.Generator.normal(loc=0.0, scale=0.5, size=len(stock_list))
#     return poids


def portfolio_simulation(stocks, start, end, nb_sim):
    returns = []
    risk = []
    s_ratio = []
    poids_list = []

    moy_return = calc_stock_returns(stocks, start, end)[0]
    cov_matrix = calc_stock_returns(stocks, start, end)[1]

    for i in range(nb_sim):
        poids_sim = poids_random(stocks)
        returns.append(perf_portfolio(poids_sim, moy_return, cov_matrix)[0])
        risk.append(perf_portfolio(poids_sim, moy_return, cov_matrix)[1])
        s_ratio.append(-sharp_ratio_opp(poids_sim, moy_return, cov_matrix))
        poids_list.append(poids_sim)

    data_sim = {"returns": returns, "risque": risk, "sharpe_ratio": s_ratio}

    for counter, symbol in enumerate (stocks):
            # print(counter, symbol)
        data_sim[symbol + " poids"] = [w[counter] for w in poids_list]

    portefeuilles_sim = pd.DataFrame(data_sim)
    print(portefeuilles_sim.head())

    #Identification du portefeuille à risque minimum:
    min_risk_port = portefeuilles_sim.iloc[portefeuilles_sim["risque"].idxmin ()]
    max_s_ratio_port = portefeuilles_sim.iloc[portefeuilles_sim["sharpe_ratio"].idxmax ()]
    print("Voici le portefeuille à risque minimum: ")
    print(min_risk_port)
    print(" ")
    print("Voici le portefeuille à ratio de Sharpe maximum: ")
    print(max_s_ratio_port)

    # Plot frontière efficace
    plt.subplots (figsize=(20, 20))
    plt.scatter (portefeuilles_sim["risque"], portefeuilles_sim['returns'], c=portefeuilles_sim["sharpe_ratio"], marker='o', s=10, alpha=0.3)
    plt.scatter(min_risk_port[1], min_risk_port[0], color='r', marker='*', s=500)
    plt.scatter(max_s_ratio_port[1], max_s_ratio_port[0], color='g', marker='*', s=500)
    plt.show ()

    return


def variance_port(poids, moy_return, cov_matrix):
    return perf_portfolio(poids, moy_return, cov_matrix)[1]


def sharp_ratio_opp(poids, moy_return, cov_matrix, ss_risque=0):
    """ Calcul du Sharp ratio (négatif ! - voir scipy) d'un portefeuille pour une pondération donnée
    INPUT:
        - poids: np.array contenant la pondération des actions
        - stock_return: array contenant la moyenne """

    p_ret, r_port = perf_portfolio(poids, moy_return, cov_matrix)
    sharp_ratio = (p_ret - ss_risque) / r_port
    return - sharp_ratio

def max_sharp_ratio(moy_return, cov_matrix, ss_risque=0, constraints_set=(0,1)):

    long_port = len(moy_return)
    init_guess = long_port*[1./long_port]
    args = (moy_return, cov_matrix, ss_risque)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraints_set
    bounds = tuple(bound for asset in range(long_port))
    result = scy.minimize(sharp_ratio_opp, init_guess, args=args, method='SLSQP',
                                     bounds=bounds, constraints=constraints)

    poids_result_df = pd.DataFrame(result['x'], index=moy_return.index, columns=['poids'])
    #print(round(poids_result_df*100,2))
    return result, poids_result_df

def minimum_variance(moy_return, cov_matrix, constraints_set=(0,1)):
    long_port = len(moy_return)
    init_guess = long_port*[1./long_port]
    args = (moy_return, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraints_set
    bounds = tuple(bound for asset in range(long_port))
    result = scy.minimize(variance_port, init_guess, args=args, method='SLSQP',
                           bounds=bounds, constraints=constraints)

    result_df = pd.DataFrame (result['x'], index=moy_return.index, columns=['poids'])


    return result, result_df

# MAIN-------------------------------------------------------------------------
stock_list = ["MSFT", "IBM", "META", "GOOG", "V", "JNJ"]
start_d = dt.datetime(2020, 1, 1)
stop_d = dt.datetime(2020, 12, 31)

price_matrix = fetch_stock_price(stock_list, start_d, stop_d)
result_calc = calc_stock_returns(stock_list, start_d, stop_d)

#print(price_matrix)
#print(result_calc)
#print(round(result_calc[0]*252*100, 2))


#weights = poids_random_stardardised(stock_list)
#port_perf = perf_portfolio(weights, result_calc[0], result_calc[1])
#print("poids: : ", weights)
#print("type port_perf: ", port_perf)
#print(type(port_perf))


#
# Optimisation function
# ----------------------
# minimi_var = minimum_variance(result_calc[0], result_calc[1])
# print(type(minimi_var))
# print("minimisation variance: ", minimi_var)
# print("Poids Optimisation Var:")
# print("optimisation Var - poids: ", minimi_var[1]*100)
# print("Perf Portfolio Minimimum Var", np.dot(result_calc[0], minimi_var[1])*252*100, " %")

# optimi_SR = max_sharp_ratio(result_calc[0], result_calc[1])
# print(type(optimi_SR))
# print("optimisation SR :", optimi_SR[0])
# print("Poids Optimisation SR:")
# print("optimisation SR - poids: ", optimi_SR[1]*100)
# print("Perf Portfolio Optimisation SR", np.dot(result_calc[0],optimi_SR[1])*252*100," %")

# Simulation porfolio
#---------------------
portfolio_simulation(stock_list, start_d, stop_d, 20000)

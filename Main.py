import datetime as dt
import scipy.optimize as scy
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def calc_stock_data(stocks, start_date, end_date):

    """
    crée un dataframe contenant les prix des actions (close price)

    :param stocks : liste (of strings) contenant les actions du portefeuille
    :param start_date : date de début
    :param end_date : date de fin

    :return returns : array contenant la moyenne des retuns journaliers des actions
    :return matrix_cov : array contenant la matrice var/cov des actions
    :return stock_price : dataframe contenant les prix des actions
    """

    stock_price = yf.download(stocks, start=start_date, end=end_date, progress=True)
    stock_price = stock_price['Close']

    log_returns = np.log(stock_price / stock_price.shift (1))
    returns = log_returns.mean()
    matrix_cov = log_returns.cov()

    return returns, matrix_cov, stock_price

def perf_portfolio(poids, returns, cov_matrix):

    """
    Calcul la performance annuelle et le risque d'un portefeuille en tenant compte de la pondération individuelle
    des actions

    :param poids: array/DF contenant les pondérations du portefeuille
    :param returns: array/DF contenant le return journalier moyen des actions
    :param cov_matrix: array/DF contenant la matrice var/cov des returns journaliers

    :return return_port: un float égale au return annuel du portefeuille
    :return risk_port: un float égale au risque du return annuel du portefeuille
    """

    return_port = np.sum(returns * poids) * 252
    risk_port = np.dot(np.dot(cov_matrix, poids),poids)**(1/2) * np.sqrt(252)
    return return_port, risk_port
#
def poids_random(stocks):
    """
    Crée un np.array avec des pondérations aléatoires standardisées (somme = 1)
    :param stocks: liste (string) contenant les actions du portefeuille

    :return poids: array contenant les pondérations aléatoires standardisées
    """

    poids = np.random.random(len(stocks))
    poids /= np.sum(poids)
    return poids

def portfolio_mc_simulation(stocks, start, end, nb_sim):
    """
    simulation de Monte Carlo: réaliser une simulation Monte-Carlo afin de sonder l'espace retour-volatilité pour
    des portefeuilles composés d'actions.

    - La fonction identifie le portefeuille ayant le ratio de Sharp maximum dans l'espace risque/return
    - La fonction identifie le portfeuille auant le risque minimum dans l'espace risque/return
    - La fonction dessine un scatter plot de chaque simulation dans l'espace risque/return

    :param stocks : liste (of strings) contenant les actions du portefeuille
    :param start : date de début (utilise le module datetime as dt)
    param end_date : date de fin (utilise le module datetime as dt)
    :param nb_sim : nombre de simulations

    :return NONE
    """

    returns = []
    risk = []
    s_ratio = []
    poids_list = []

    moy_return = calc_stock_data(stocks, start, end)[0]
    cov_matrix = calc_stock_data(stocks, start, end)[1]

    for i in range(nb_sim):
        poids_sim = poids_random(stocks)
        returns.append(perf_portfolio(poids_sim, moy_return, cov_matrix)[0])
        risk.append(perf_portfolio(poids_sim, moy_return, cov_matrix)[1])
        s_ratio.append(-1 * sharp_ratio_opp(poids_sim, moy_return, cov_matrix))
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

    #Plot Simulation:

    plt.subplots (figsize=(20, 20))
    plt.scatter (portefeuilles_sim["risque"], portefeuilles_sim['returns'], c=portefeuilles_sim["sharpe_ratio"],
                 cmap="viridis", marker='o', s=10, alpha=0.3)
    plt.scatter(min_risk_port[1], min_risk_port[0], color='r', marker='*', s=500)
    plt.scatter(max_s_ratio_port[1], max_s_ratio_port[0], color='g', marker='*', s=500)
    plt.show ()

    return

def variance_port(poids, moy_return, cov_matrix):
    """
    Cette fonction est utilisée uniquement dans le cadre de la fonction d'optimisation 'minimum_variance'

    :param poids: array/DF contenant les pondérations du portefeuille
    :param moy_return: array/DF contenant le return journalier moyen des actions
    :param cov_matrix: array/DF contenant la matrice var/cov des returns journaliers

    :return: renvoie le risque du portfeuille (retun de la fonction perfportfoilio[1))
    """

    return perf_portfolio(poids, moy_return, cov_matrix)[1]

def sharp_ratio_opp(poids, returns, cov_matrix, ss_risque=0):
    """
    Calcul du Sharp ratio (négatif ! afin d'utiliser la fonction scipy.optimize.minimize de la librairie SciPy)
    d'un portefeuille pour une pondération donnée.

    :param poids: np.array contenant la pondération des actions
    :param return: array contenant la moyenne

    :return -sharp_ratio contenant la valeur negative du sharp ratio
    """

    p_ret, r_port = perf_portfolio(poids, returns, cov_matrix)
    sharp_ratio = (p_ret - ss_risque) / r_port
    return - sharp_ratio

def max_sharp_ratio(returns, cov_matrix, ss_risque=0, constraints_set=(0,1)):
    """
    Cette fonction calcule le portefeuille avec le ratio de Sharp maximum en utilisant un algorythme de minimisation sous
    contraintes de la librairie SciPY.

    :param returns: array/DF contenant le return journalier moyen des actions
    :param cov_matrix: array/DF contenant la matrice var/cov des returns journaliers
    :param ss_risque: taux sans risque du marché (fixé à zero)
    :param constraints_set: les bornes pour les variables d'optimisation

    :return: none
    """

    long_port = len(returns)
    init_guess = long_port*[1./long_port]
    args = (returns, cov_matrix, ss_risque)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraints_set
    bounds = tuple(bound for asset in range(long_port))
    result = scy.minimize(sharp_ratio_opp, init_guess, args=args, method='SLSQP',
                                     bounds=bounds, constraints=constraints)

    poids_result_df = pd.DataFrame(result['x'], index=returns.index, columns=['poids'])
    ret_port_opt = np.dot (returns, poids_result_df["poids"]) * 252 * 100

    print("Optimisation réalisée avec Sharp Ratio optimisé = ",-result.fun)
    print ("Poids optimisation - Sharp Ratio :", poids_result_df * 100)
    print ("Return Portfolio Optimisation Sharp Ratio", round(ret_port_opt,2), " %")

    return

def minimum_variance(returns, cov_matrix, constraints_set=(0,1)):
    """
    Cette fonction calcule le portefeuille avec le risque minimum en utilisant un algorythme de minimisation sous
    contraintes de la librairie SciPY.

    :param returns: array/DF contenant le return journalier moyen des actions
    :param cov_matrix: array/DF contenant la matrice var/cov des returns journaliers
    :param constraints_set: les bornes pour les variables d'optimisation

    :return: none
    """

    long_port = len(returns)
    init_guess = long_port*[1./long_port]
    args = (returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraints_set
    bounds = tuple(bound for asset in range(long_port))
    result = scy.minimize(variance_port, init_guess, args=args, method='SLSQP',
                           bounds=bounds, constraints=constraints)

    poids_result_df = pd.DataFrame (result['x'], index=returns.index, columns=['poids'])
    ret_port_opt = np.dot (returns, poids_result_df["poids"]) * 252 * 100

    print ("Optimisation réalisée. Risque du portefeuille:  ", result.fun)
    print ("Poids des actions pour Risque Minimum: ", poids_result_df * 100)
    print ("Return du portefeuille de Risque Minimum ", round (ret_port_opt, 2), " %")
    print ("Sharp Ratio du portefeuille de Risque Minimum: ", -sharp_ratio_opp(poids_result_df["poids"], returns, cov_matrix))

    return

def random_walk(stocks, start_date, end_date, nb_sim, nb_walk):
    """
    Cette fonction va exécuter un Random Walk sur les prix des actions pour une période de 252 jours en partant du
    dernier prix connu de chaque action. Elle va ensuite exécuter une simulation de Monte Carlo sur base des prix
    simulés pour chaque action.
    Sur base de cette dernière simulation, la fonction identifie:

        - le portefeuille ayant le ratio de Sharp maximum dans l'espace risque/return
        - le portfeuille auant le risque minimum dans l'espace risque/return

    La fonction dessine enfin un scatter plot de chaque simulation dans l'espace risque/return

    :param stocks: liste (of strings) contenant les actions du portefeuille
    :param start_date: date de début
    :param end_date:
    :param nb_sim: nombre de simulation de Monte Carlo pour le portefeuille
    :param nb_walk: nombre de random walk pour chaque action
    :return:
    """

    data_sim_price = pd.DataFrame(columns=stocks)
    price_df = calc_stock_data(stocks, start_date, end_date)[2]
    print(price_df.iloc[-1])

    for asset in stocks:
        returns = np.log(1+price_df[asset].pct_change())
        mu, sigma = returns.mean(), returns.std()
        sim_price_arr = np.empty([252])
        asset_price_arr = np.empty([252])

        for i in range(nb_walk):
            sim_ret = np.random.normal(mu, sigma, 252)
            initial = price_df[asset].iloc[-1]
            sim_price_arr = initial * (sim_ret + 1).cumprod()
            asset_price_arr = np.add(asset_price_arr, sim_price_arr)

        asset_price_arr = asset_price_arr/nb_walk
        data_sim_price[asset] = asset_price_arr

    print("Nouvelle matrice de prix des actions après une random walk de 252 jours): ")
    print(data_sim_price)
    print("")
    print("Simulation de Monte Carlo avec ces nouveaux prix: ")

    returns = []
    risk = []
    s_ratio = []
    poids_list = []
    log_returns = np.log (data_sim_price / data_sim_price.shift(1))
    moy_return = log_returns.mean()
    cov_matrix = log_returns.cov()

    for i in range (nb_sim):
        poids_sim = poids_random (stocks)
        returns.append (perf_portfolio (poids_sim, moy_return, cov_matrix)[0])
        risk.append (perf_portfolio (poids_sim, moy_return, cov_matrix)[1])
        s_ratio.append (-1 * sharp_ratio_opp(poids_sim, moy_return, cov_matrix))
        poids_list.append (poids_sim)

    data_sim = {"returns": returns, "risque": risk, "sharpe_ratio": s_ratio}

    for counter, symbol in enumerate (stocks):
        # print(counter, symbol)
        data_sim[symbol + " poids"] = [w[counter] for w in poids_list]

    portefeuilles_sim = pd.DataFrame (data_sim)
    print (portefeuilles_sim.head ())

    # Identification du portefeuille à risque minimum:
    min_risk_port = portefeuilles_sim.iloc[portefeuilles_sim["risque"].idxmin ()]
    max_s_ratio_port = portefeuilles_sim.iloc[portefeuilles_sim["sharpe_ratio"].idxmax ()]
    print ("Voici le portefeuille à risque minimum: ")
    print (min_risk_port)
    print (" ")
    print ("Voici le portefeuille à ratio de Sharpe maximum: ")
    print (max_s_ratio_port)

    # Plot Simulation:

    plt.subplots (figsize=(20, 20))
    plt.scatter (portefeuilles_sim["risque"], portefeuilles_sim['returns'], c=portefeuilles_sim["sharpe_ratio"],
                 cmap="viridis", marker='o', s=10, alpha=0.3)
    plt.scatter (min_risk_port[1], min_risk_port[0], color='r', marker='*', s=500)
    plt.scatter (max_s_ratio_port[1], max_s_ratio_port[0], color='g', marker='*', s=500)
    plt.show ()

    return

# MAIN-------------------------------------------------------------------------
stock_list = ["MSFT", "IBM", "META", "GOOG", "V", "JNJ", "PG"]
start_d = dt.datetime(2020, 1, 1)
stop_d = dt.datetime(2020, 12, 31)
result_calc = calc_stock_data(stock_list, start_d, stop_d)

# Test Random Walk
#------------------
#random_walk(stock_list, start_d, stop_d, 40000, 100)
#
# Test Optimisation function
# ----------------------

#Test minimum_variance
#---------------------
#minimi_var = minimum_variance(result_calc[0], result_calc[1])
# print(type(minimi_var))
# print("minimisation variance: ", minimi_var)
# print("Poids Optimisation Var:")
# print("optimisation Var - poids: ", minimi_var[1]*100)
# print("Perf Portfolio Minimimum Var", np.dot(result_calc[0], minimi_var[1])*252*100, " %")


#Test max_sharp_ratio
#--------------------
#optimi_SR = max_sharp_ratio(result_calc[0], result_calc[1])


# Test Simulation Monte Carlo
portfolio_mc_simulation(stock_list, start_d, stop_d, 20000)

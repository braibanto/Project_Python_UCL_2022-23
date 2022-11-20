# Project_Python_UCL_2022-23 (Frontière optimum pour Portefeuille actions)

Folder for the LPHYS Python project 2022-23

Calculer le log retour journalier de votre action comme

  logRi = j∑ pj log(Ri+1 / Ri).

avec
  j∑ est la somme sur chaque élément du portefeuille.
  pj est le poids d'une action dans le portefeuille.
  Ri est la valeur de l'action au jour iii à la fermeture (AdjCloseAdj CloseAdjClose dans pandas).

Estimer le log retour annuel moyen (attention une année en finance dure 252 jours), la volatilité annuelle et calculer le ratio de Sharp défini comme

  SR=Rp−Rf / σp

avec
  Rp est le retour du portefeuille, il est estimé ici comme étant le log retour annuel.
  Rf est le retour d'un investissement sans risque (ici on peut le fixer à 0).
  σp est la volatilité annuelle du portefeuille.

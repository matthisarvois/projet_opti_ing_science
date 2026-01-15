import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Modèles de régression
from sklearn.linear_model import LinearRegression

# Modèles de classification


#On charge les données qui sont dans les packages de python
diabetes = datasets.load_diabetes()

X = diabetes.data      # variables explicatives (p ~ 10)
y = diabetes.target    # variable réponse continue

# On sépare en échantillon d'apprentissage et de test de manière aléatoire
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=890)

#Le premier modèle consiste à faire une régression linéaire, alors voici un OLS simple.
ols = LinearRegression()

# On ajuste le modèle sur les données d'entraînement (pour pouvoir les tester sur les autres donnés apres)
ols.fit(X_train, y_train)

# On essaye de prédire y via l'échantillon de test
y_pred = ols.predict(X_test)

# Puis à la fin on regarde l'erreur quadratique sur le test que nous venon de réaaliser
mse_test = mean_squared_error(y_test, y_pred)

print("MSE (test) - Régression linéaire :", mse_test)

e = y_test - y_pred
h = np.mean(e)

print(f"La moyenne des erreur est de {h}")

#On peut regarder la différence entre la y pred et le y test en visuel;
plt.plot(e)
plt.show()



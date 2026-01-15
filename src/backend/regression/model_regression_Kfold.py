import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error

# Modèles de régression
from sklearn.linear_model import LinearRegression


#On prend le même commencement que les méthodes OLS
diabetes = datasets.load_diabetes()
X = diabetes.data      
y = diabetes.target    
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
ols = LinearRegression()
ols.fit(X_train, y_train)
y_pred = ols.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)

#Cette fois si on va faire une méthode de validation croisée.
#On choisit K = 10 groupes
kfold = KFold(n_splits=10, shuffle=True, random_state=890)

# On évalue le modèle OLS par validation croisée ,comme dans le cours on fait de la negative mean
scores = cross_val_score(
    LinearRegression(),
    X, y,
    cv=kfold,
    scoring="neg_mean_squared_error"
)

# On remet le signe + pour avoir un MSE moyen
mse_cv = -scores.mean()
print("MSE moyen (10-fold CV) - OLS :", mse_cv)

#On, voit qu'avec une méthode de validation croisée notre erreur diminue d'à peu près 600 ce qui est non négligeable.
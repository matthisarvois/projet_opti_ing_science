import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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


#On regardes mes choses de manière aléatoire ici
rng = np.random.RandomState(890)
n_boot = 500              # nombre de rééchantillonnages bootstrap mis à 500
n, p = X_train.shape        #On prend les cooronnées de nos données

coefs_boot = np.zeros((n_boot, p))  # pour stocker les coefficients  du bootstrap
intercepts_boot = np.zeros(n_boot)

for b in range(n_boot):     # Ici on fait une bouble sur le nombre de boot initialisé (ici 500)
    indices = rng.randint(0, n, size=n) # On prend indices parmi toutes les lignes d'entrainement
    
    #Ici on créer les nouveaux groupe sur lesquels on va faire la regression.
    X_b = X_train[indices] 
    y_b = y_train[indices]
    
    # On ajuste un modèle OLS sur cet échantillon bootstrapé
    ols_b = LinearRegression()
    ols_b.fit(X_b, y_b)
    intercepts_boot[b] = ols_b.intercept_
    
    # On stocke les coefficients
    coefs_boot[b, :] = ols_b.coef_

# Moyenne et écart-type bootstrap des coefficients, comme ca on a les coefficients pour pouvoir les réintégrer et regarder l'erreur
coefs_mean = coefs_boot.mean(axis=0)
coefs_std = coefs_boot.std(axis=0)
intercept_mean = intercepts_boot.mean()
#Maintenant on peut essayer 

y_hat_boot = intercept_mean + X_test @ coefs_mean #Attention à ne pas oublier l'intercept qui jou un grôle important.
mse_test_boot = mean_squared_error(y_test,y_hat_boot)


print("Moyenne bootstrap des coefficients :", coefs_mean)
print("Écart-type bootstrap des coefficients :", coefs_std)
print("Finalementl'erreur sur l'échantillon test est : ", mse_test_boot)

#Finalemnt nous avons encore gagner en précision en utilisant un algorithme boot car notre rendu a donné un MSE à 200 de moins que
# le Kfold soit 800 de moins que la regression simple

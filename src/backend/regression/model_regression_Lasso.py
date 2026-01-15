import numpy as np
import matplotlib.pyplot as plt 

from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, accuracy_score

# Modèles de régression
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Modèles de classification
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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

#dans ce fichier on va voir comment ont fait une méthode Lasso

#On fait exactement la même chose sauf que dans le make_pipeline on fait
# un lasso au lieu de Riges, On rajoute le alpha qui diffère légèrement
lasso = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.1, max_iter=10000)
)


#On fait un .fit comme toutes les autres méthodes
lasso.fit(X_train, y_train)

#Un predict comme toutes les autres méthodes
y_pred_lasso = lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)


#et finalement on regarde en fraisant un printe du MSE
print("MSE (test) - Lasso (λ = 0.1) :", mse_lasso)

# On peut inspecter le nombre de coefficients non nuls
# (on récupère l'étape Lasso dans le pipeline)
lasso_coef = lasso.named_steps["lasso"].coef_
print("Nombre de coefficients non nuls (Lasso) :", np.sum(lasso_coef != 0))

##Possibilité de faire une séléction du meileur chiffre ? 

val = []
list = np.linspace(0.001,10,100)

for l in list :
    lasso = make_pipeline(
    StandardScaler(),
    Lasso(alpha=l, max_iter=10000)
)


    #On fait un .fit comme toutes les autres méthodes
    lasso.fit(X_train, y_train)

    #Un predict comme toutes les autres méthodes
    y_pred_lasso = lasso.predict(X_test)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    
    val.append(mse_lasso)

plt.plot(list,val)
plt.show()

#Ici on a un minimum qui tourne autour de 1,5, ainsi on peut écrire que .

best_lam = list[np.argmin(val)]
good_val = val[np.argmin(val)]

print(f"Le meilleur lambda est {best_lam}, avec une erreur à {good_val}")
#Ce modèles est le meileur parmi ceux qui traitent de la regression.



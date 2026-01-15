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


#On va faire dans ce fichier une méthode de regression par Ridge
# On standardise les X AVANT la pénalisation (important pour Ridge et Lasso)
#Et on va faire une boucle sur 3 chiffres différents comme ca on saura quel alpha prendre.
al = np.linspace(1,100,1000)
val = []
for alph in al:
    
    ridge = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alph)   #On met le alpha à 1 car dans le cours, le professeur l'a mis ainsi
    )

    ridge.fit(X_train, y_train) #Ici on créer le vecteur

    y_pred_ridge = ridge.predict(X_test)#ici on créer l'estimateur par les moyens de riges
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)

    val.append(mse_ridge)
    
#On voit une légère convergence quand au choix du alpha.
#On peu regarder grahiquement

plt.plot(al,val)
plt.show()


least = min(val)
print(f"Le paramètre optimal est 50 pour une erreur à {least}")


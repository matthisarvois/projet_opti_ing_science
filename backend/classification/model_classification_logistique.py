"""
Nous allons dans ce fichier tester de niombreuse méthode de prédiction en utilisant des méthodes d'apprentissage automatique supervisé

Ces méthodes et ces tests visent à l'amélioration des connaissances et des compétences que j'ai acquises en vu de faire un projet personel que je mennerias dans
quelques temps.
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

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

#pour tous les modèles de classification qui se ferons dans cette partie, nous utiliserons le mêmes jeu de données
#avec les mêmes variables intégrées dans des packages 

cancer = datasets.load_breast_cancer()

Xc = cancer.data
yc = cancer.target  # 0 = malin, 1 = bénin

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.3, random_state=42, stratify=yc
)

#Comme pour les méthodes de regression, ici nous faisant une étape de standardisation afin d'optimiser les modèles.
log_reg = make_pipeline(
    StandardScaler(),  # important, variables à échelles différentes
    LogisticRegression(max_iter=10_000) #Fonction pour faire la regression logisqtique ; LogisticRegression()
)

#Comme d'habitude on fit les données trouvée et on les .predict 
#(qy'on met dans une variable yc_pred  en vu de tester les différences avec les vraies données)
log_reg.fit(Xc_train, yc_train)
yc_pred = log_reg.predict(Xc_test)

#accuracy_score est le score de ressemblence entre la variable test et la variable 
acc = accuracy_score(yc_test, yc_pred)

print("Accuracy (test) - Régression logistique :", acc)


#Puis on fait une petite méthode de valisation croisée pour optimisées le résultat de la classification
scores = cross_val_score(
    log_reg, Xc, yc,
    cv=10,
    scoring="accuracy"
)
print("Accuracy moyenne (10-fold CV) - RL :", scores.mean())

def classi_logi(df,target):
    X = df.drop(columns=target).to_numpy()
    y = df[target].to_numeric()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.3, stratify=y
    )
    list = [X_train, X_test, y_train, y_test]
    #creation du model
    cla = make_pipeline(
        StandardScaler(),
        LogisticRegression()
    )
    
    return cla, list
    
    



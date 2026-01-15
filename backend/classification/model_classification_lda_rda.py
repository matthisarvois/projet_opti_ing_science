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


cancer = datasets.load_breast_cancer()

Xc = cancer.data
yc = cancer.target  # 0 = malin, 1 = bénin

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.3, random_state=42, stratify=yc
)
#### Analyse lda

#On fait en réalité toujours la même chose, simplement nous changeons la fonction dans make_pipeline()
lda = make_pipeline(
    StandardScaler(),
    LinearDiscriminantAnalysis()
)

lda.fit(Xc_train, yc_train)
yc_pred_lda = lda.predict(Xc_test)
print("Accuracy (test) - ADL :", accuracy_score(yc_test, yc_pred_lda))

#### Analyse qda

qda = make_pipeline(
    StandardScaler(),
    QuadraticDiscriminantAnalysis()
)

qda.fit(Xc_train, yc_train)
yc_pred_qda = qda.predict(Xc_test)
print("Accuracy (test) - ADQ :", accuracy_score(yc_test, yc_pred_qda))

#j'aui compris et il faut en réalité s'aproché du 1. Ainsi cela fait un autre facteur de choix pour les modèles
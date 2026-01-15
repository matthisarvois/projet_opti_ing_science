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
##analyse via random forest

#Petite différence est que la on utilise directement une fonction RandomForestClassifier()
rf = RandomForestClassifier(
    n_estimators=200,      # nombre d'arbres
    max_depth=None,       # profondeur max (None = jusqu'à pureté)
    random_state=42
)

rf.fit(Xc_train, yc_train)
yc_pred_rf = rf.predict(Xc_test)
print("Accuracy (test) - Random Forest :", accuracy_score(yc_test, yc_pred_rf))

# Validation croisée encore et toujours
scores = cross_val_score(
    rf, Xc, yc,
    cv=10,
    scoring="accuracy"
)

#et on affiche le resultat.
print("Accuracy moyenne (10-fold CV) - Random Forest :", scores.mean())
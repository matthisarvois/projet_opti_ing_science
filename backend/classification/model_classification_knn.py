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

### Analyse knn pour regarder les types de populations.
#D'abord avec un k fix
knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

knn.fit(Xc_train, yc_train)
yc_pred_knn = knn.predict(Xc_test)
print("Accuracy (test) - KNN (k=5) :", accuracy_score(yc_test, yc_pred_knn))

#Puis avec un k choisi par validation croisée.
k_values = range(1, 21)
acc_by_k = []

for k in k_values:
    knn_k = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k)
    )
    scores = cross_val_score(
        knn_k, Xc, yc,
        cv=10,
        scoring="accuracy"
    )
    acc_by_k.append(scores.mean())

best_k = k_values[int(np.argmax(acc_by_k))]
print("Meilleur k selon CV :", best_k)







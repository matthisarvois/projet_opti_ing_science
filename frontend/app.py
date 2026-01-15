# app.py
import streamlit as st
from sklearn import datasets
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from backend.engineering.load_dataset import main_load

data_frame = main_load()

st.set_page_config(
    page_title="Choose Model",
    layout="wide",
)

# === BARRE LATERALE (NAVIGATION) ===
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Aller à :",
    ("Analyse des données", "Explications de variables", "Prédiction"),
)

st.sidebar.markdown("---")


# ===================== PAGE : ANALYSE DES DONNÉES =====================
if page == "Analyse des données":
    st.title("Analyse des données")
    page_analyse = st.selectbox("Navigation dans l'analyse des données :",
                                    ("Importer les données",
                                     "Analyse univariée",
                                     "Analyse bivariée",
                                     "Analyse multivariée"))
    #=Sous page de l'analyse : import des données=#
    if page_analyse=="Importer les données" :
        st.write("Prochainement nous ferons un sorte que toutes les bases de données puissent être étudiées, mais pour l'instant nous ferons des analyses sur la base de données sur le diabète")
        # Placeholders pour plus tard
        #ici je vais faire une page qui prend une fonction
    if page_analyse=="Analyse bivariée" :
        st.write("Dans cette page nous étudierons touts les liens possibles entre les variables selons les bases de données")
        st.markdown("---")
        #On fait une selectbox sur les différents noms des datasets.
        #fonction qui load des datasets s'il y en a
        #data = st.selectbox()
        
        ## Pour l'instant on load le cancer dans le début de l'app.
        list_columns = list(data_frame.columns)
        list_var_comp = list(data_frame.columns)
        
        st.info("Dans cette partie vous allez choisir quelles variables vous voudriez comparer et étudier.")
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            var_1 = st.selectbox("Première variable à comparer",list_columns)
        with col2:
            var_2 = st.selectbox("Deuxième variable à comparer",list_var_comp)
            
        type_var_1 = data_frame[var_1].dtypes
        type_var_2 = data_frame[var_2].dtypes
        
        st.markdown("---")
        
        if var_1 == var_2:
            st.warning("Attention les variables séléctionnées sont identiques...😅")
        else :
            #je fais faire juste un graphique dans le cas ou j'ai deux floats
            col1_bivar,col2_bivar = st.columns(2)
            
            with col1_bivar:
                st.markdown("## Affichage simple des points")
                fig_scatter = px.scatter(data_frame,x = var_1, y = var_2, trendline="ols")
                st.plotly_chart(fig_scatter)
            with col2_bivar:
                corr = data_frame[var_1].corr(data_frame[var_2])
                corr_pct = abs(round(corr*100,2))
            
                st.metric(label="Corrélation (%)",value=f"{corr_pct} %")
        st.markdown("## Ajouter une ou plusieurs légendes")
                
        

        
    if page_analyse=="Analyse univariée" :
        st.write("Dans cette page nous étudirons toutes les variables unes à unes")
    
    if page_analyse=="Analyse multivariée" :
        st.write("Cette page aura pour objectif d'étudier les liaisons entre les variables de manières groupées")


# ===================== PAGE : EXPLICATIONS DE VARIABLES =====================
elif page == "Explications de variables":
    st.title("📚 Explications de variables")
    st.write("Ici tu expliques les variables, leurs types, leurs rôles, etc.")

    # Exemples de placeholders
    st.subheader("Dictionnaire de variables")
    st.info("➡️ Tu pourras afficher un tableau avec nom, type, description.")

    st.subheader("Notes")
    st.info("➡️ Tu pourras ajouter du texte libre pour commenter les variables.")


# ===================== PAGE : PREDICTION =====================
else:
    st.title("📈 Prédiction")

    # --- 1) Choix du modèle ---
    st.header("1️⃣ Choix du modèle")

    type_probleme = st.selectbox(
        "Type de problème",
        ["Classification", "Régression"],
    )

    modele = None
    if type_probleme == "Classification":
        modele = st.selectbox(
            "Modèle de classification",
            [
                "Logistic Regression",
                "Random Forest Classifier",
                "SVM",
                "k-NN",
            ],
        )
    else:
        modele = st.selectbox(
            "Modèle de régression",
            [
                "Linear  Regression",
                "Random Forest Regressor",
                "Ridge",
                "Lasso",
            ],
        )

    st.write(f"✅ Modèle sélectionné : **{modele}**")

    st.markdown("---")

    # --- 2) Entraînement du modèle ---
    st.header("2️⃣ Entraînement du modèle")

    st.caption("Ici tu mettras les options d’entraînement (hyperparamètres, train/test split, etc.).")

    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider(
            "Taille du test (proportion)",
            min_value=0.1,
            max_value=0.5,
            value=0.3,
            step=0.05,
        )

        random_state = st.number_input(
            "Random state",
            min_value=0,
            value=42,
            step=1,
        )

    with col2:
        st.write("Hyperparamètres (exemple)")
        # Ici tu pourras adapter selon le modèle
        n_estimators = st.slider(
            "n_estimators (si Random Forest)",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
        )

    st.markdown("---")

    lancer_train = st.button("🚀 Entraîner le modèle")

    if lancer_train:
        st.info("➡️ Ici tu appelleras ta fonction d’entraînement avec les paramètres ci-dessus.")
        # Exemple de ce que tu feras plus tard :
        # model = build_model(modele, n_estimators=..., ...)
        # X_train, X_test, y_train, y_test = train_test_split(..., test_size=test_size, random_state=random_state)
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # metrics = compute_metrics(...)
        st.success("Squelette d’entraînement OK – à connecter à ton code métier.")


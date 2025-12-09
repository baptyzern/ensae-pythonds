#les librairies

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

#La classe


class pipeline_modelisation :

    """
    Présentation générale
    --------------------

    Ce module implémente une classe Python dédiée à la modélisation statistique flexible,
    inspirée des travaux de Lino Galiana et Alexandre Dore. 
    Cette approche, structurée, reproductible et centrée sur les bonnes pratiques
    en science des données, sert de fil conducteur à l’ensemble du travail.

    Objectifs
    ---------
    - Préparer et transformer les données
    - Sélectionner les variables pertinentes
    - Estimer le modèle (OLS, Lasso, Ridge)
    - Valider statistiquement et diagnostiquer les résidus

    Pour plus de détails, consulter :
    - https://pythonds.linogaliana.fr
    - https://doi.org/10.5281/zenodo.8229676

    Méthodes principales
    -------------------

    1. Initialisation (__init__)
    ------------------------
    Crée la structure de base du modèle, prend en entrée :
    - la base de données,
    - les ensembles de variables explicatives,
    - la variable cible.
    Pose les fondations du pipeline de modélisation.

    2. Prétraitement des variables (preprocessing_features)
    -----------------------------------------------------
    Standardise et prépare les données :
    - Transformation logarithmique de la cible (si cible_transform="log") :
        y_trans = log(y) ou y sinon
    - Création de variables indicatrices (one-hot)
    - Standardisation des variables numériques :
        x_j_scaled = (x_j - μ_j) / σ_j

    3. Sélection des variables explicatives (features_selection)
    ----------------------------------------------------------
    Sélection automatique via Lasso :
    Minimisation : 
        hat{β}^Lasso = argmin_beta { (1/2n)||y-Xβ||_2^2 + λ||β||_1 }
    - y ∈ ℝ^n : variable cible
    - X ∈ ℝ^{n×p} : variables explicatives prétraitées
    - λ > 0 : paramètre de régularisation
    Les variables dont β_j = 0 sont éliminées.

    4. Visualisation de l’importance des variables (features_viz)
    -----------------------------------------------------------
    Affiche les coefficients estimés par le Lasso :
    - valeurs positives/négatives
    - variables éliminées (β_j = 0)
    Utile pour communiquer les résultats et valider l’intelligibilité.

    5. Choix du paramètre de régularisation (penalization_choice_curve)
    ----------------------------------------------------------------
    Reproduit la courbe de validation croisée pour différentes λ :
    - Lasso (L1) : hat{β}(λ) = argmin_beta { (1/2n)||y-Xβ||_2^2 + λ||β||_1 }
    - Ridge (L2) : hat{β}(λ) = argmin_beta { (1/2n)||y-Xβ||_2^2 + λ||β||_2^2 }
    Affiche le MSE moyen par fold pour choisir λ optimal.

    6. Ajustement du modèle final (Model)
    -----------------------------------
    Options :
    - Modèle linéaire classique (OLS) : y = Xβ + ε, ε ~ N(0, σ^2 I_n)
    - Estimation robuste (HC0-HC3)
    - Régularisation (Lasso ou Ridge)
    7. Diagnostic, validation et prédiction (summarize, residuals_validation & predict)
    -----------------------------------------------------------------------------------
    - Résumé complet du modèle : coefficients, erreurs-types, p-values, R², AIC, etc. (`summarize`)
    - Analyse des résidus : distribution, normalité, tendances structurelles (`residuals_validation`)
    - Prédiction des nouvelles observations (`predict`) :
        - Permet de prédire sur les données d’entraînement ou sur de nouvelles données.
        - Peut retourner les valeurs exponentiées si la cible avait été log-transformée.

    Pour plus de détails sur les étapes : 
    - Prétraitement : https://pythonds.linogaliana.fr/content/modelisation/0_preprocessing.html
    - Sélection des variables : https://pythonds.linogaliana.fr/content/modelisation/4_featureselection.html
    - Régression : https://pythonds.linogaliana.fr/content/modelisation/3_regression.html
    """
    def __init__(self , df , features , target):
        """
        df : DataFrame
        lycee_cols, ips, biblio_cols : listes de colonnes explicatives
        cible : nom de la variable cible
        """
        self.lycee_cols = features["lycee_cols"]
        self.ips = features["ips"]
        self.biblio_cols = features["biblio_cols"]
        
        # Matrice X
        self.X = df[self.lycee_cols + self.ips + self.biblio_cols].copy()
        
        # Variable cible
        self.y = df[target].copy()
        
        # Ajout de la constante

        # Objets techniques
        self.log_y = None
        self.dummies = None
        self.scaler = None
        self.X_scaled = None
        self.model = None

    def get_features(self):
        return (self.X , self.y)
    
    def preprocessing_features(self, cible_transform="none"):

        """
        - cible_transform : "none" ou "log"
        - création de dummies
        - standardisation des colonnes numériques
        """

        # Transformation de la cible

        if cible_transform.lower() == "log":
            self.log_y = np.log(self.y)
        else:
            self.log_y = self.y.copy()
        
        # Dummy encoding
        self.dummies = pd.get_dummies(self.X, drop_first=True)

        # Standardisation
        num_cols = self.dummies.select_dtypes(include=np.number).columns
        self.scaler = StandardScaler()
        self.X_scaled = self.dummies.copy()
        self.X_scaled[num_cols] = self.scaler.fit_transform(self.dummies[num_cols])

        return self.X_scaled, self.log_y


    def features_selection(self, by="lasso"):
        """
        Sélection par LassoCV (automatique)
        """
        if self.X_scaled is None:
            raise ValueError("Lance d'abord preprocessing_features()")

        if by == "lasso":
            lasso = LassoCV(cv=5, random_state=123).fit(self.X_scaled, self.log_y)
            coef = pd.Series(lasso.coef_, index=self.X_scaled.columns)
            self.selected_features = coef[coef != 0].index.tolist()
            return self.selected_features
        
        else:
            raise ValueError("Méthode non supportée : choisis 'lasso'")

    def features_viz(self):
        """
        Visualisation des coefficients Lasso
        """
        if not hasattr(self, 'selected_features'):
            raise ValueError("Lance features_selection()")

        coef = pd.Series(
            np.zeros(len(self.X_scaled.columns)),
            index=self.X_scaled.columns
        )
        # Remplace par coefficients non nuls
        lasso = LassoCV(cv=5).fit(self.X_scaled, self.log_y)
        coef = pd.Series(lasso.coef_, index=self.X_scaled.columns)

        coef.sort_values().plot(kind="barh", figsize=(8,12))
        plt.title("Importance des features (Lasso)")
        plt.show()

    def penalization_choice_curve(self, penalization="Lasso", lambdas=None, cv=5):

        """
        Choix de la régularisation par cross-validation.
        Trace la courbe log(lambda) vs MSE moyen CV.
        
        penalization : Lasso ou Ridge
        lambdas : liste ou array de valeurs de régularisation
        cv : nombre de folds pour la CV
        """

        if lambdas is None:
            lambdas = np.logspace(-4, 2, 30)  # valeurs par défaut

        selected = getattr(self, "selected_features", self.X_scaled.columns)
        X = sm.add_constant(self.X_scaled[selected]).values
        y = self.log_y.values if self.log_y is not None else self.y.values

        mean_mse = []

        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        for lam in lambdas:
            fold_mse = []
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                L1_wt = 1.0 if penalization == "Lasso" else 0.0
                model = sm.OLS(y_train, X_train).fit_regularized(L1_wt=L1_wt, alpha=lam)
                y_pred = np.dot(X_test, model.params)
                fold_mse.append(mean_squared_error(y_test, y_pred))
            mean_mse.append(np.mean(fold_mse))

        mean_mse = np.array(mean_mse)

        # Plot log(lambda) vs MSE
        plt.figure(figsize=(8,5))
        plt.plot(np.log10(lambdas), mean_mse, marker='o')
        plt.xlabel("log10(lambda)")
        plt.ylabel("MSE moyen CV")
        plt.title(f"Choix de lambda par CV ({penalization})")
        plt.grid(True)
        plt.show()

        # Meilleur lambda
        best_lambda = lambdas[np.argmin(mean_mse)]
        print(f"Meilleur lambda trouvé : {best_lambda:.5f} (MSE minimum = {mean_mse.min():.5f})")
        return best_lambda

    def fit(self, specify="ols_linear_regression", robust="False", penalization="None", best_lambda=1.0):
        """
        Construction du modèle OLS StatsModels avec options :
        - robuste (HC0, HC1, HC2, HC3)
        - pénalisation Lasso (L1) ou Ridge (L2) via fit_regularized
        alpha : force de régularisation pour Lasso/Ridge
        """
        if specify != "ols_linear_regression":
            raise ValueError("Modèle non reconnu")
        
        # Variables retenues après sélection
        selected = getattr(self, "selected_features", self.X_scaled.columns)
        X_for_model = sm.add_constant(self.X_scaled[selected])
        y_for_model = self.log_y if self.log_y is not None else self.y

        # OLS classique ou robuste
        if penalization == "None":
            ols_model = sm.OLS(y_for_model, X_for_model)
            if robust == "False":
                self.model = ols_model.fit()
            else:
                self.model = ols_model.fit(cov_type=robust)

        # Pénalisation via fit_regularized
        elif penalization in ["Lasso", "Ridge"]:
            if penalization == "Lasso":
                L1_wt = 1.0  # L1 complète → Lasso
            else:
                L1_wt = 0.0  # L2 complète → Ridge
            self.model = sm.OLS(y_for_model, X_for_model).fit_regularized(L1_wt=L1_wt, alpha = best_lambda)
        
        else:
            raise ValueError("Choix de pénalisation non reconnu : 'None', 'Lasso', 'Ridge'")

        return self.model

    def summarize(self):
        if self.model is None:
            raise ValueError("Lance Model() d'abord.")
        print(self.model.summary())

    def residuals_validation(self):
        if self.model is None:
            raise ValueError("Lance Model() d’abord")
        
        residuals = self.model.resid

        fig, axes = plt.subplots(1,2, figsize=(12,5))

        # Distribution
        sns.histplot(residuals, kde=True, ax=axes[0])
        axes[0].set_title("Distribution des résidus")

        # QQplot
        sm.qqplot(residuals, line='45', ax=axes[1])
        axes[1].set_title("QQ-plot des résidus")

        plt.tight_layout()
        plt.show()

        return residuals

        def predict(self, X_new=None, log_transform=False):
            """
            Prédit les valeurs pour de nouvelles données.
            
            X_new : DataFrame (optionnel)
                Si None, utilise les données originales X_scaled.
            log_transform : bool
                Si True, retourne les prédictions exponentiées si la cible avait été log-transformée.
            """
            if self.model is None:
                raise ValueError("Lance Model() d'abord.")
            
            if X_new is None:
                X_new_scaled = self.X_scaled
            else:
                # Si nouvelles données, appliquer même preprocessing que sur le training set
                dummies_new = pd.get_dummies(X_new, drop_first=True)
                # Conserver uniquement les colonnes présentes dans le modèle
                missing_cols = set(self.X_scaled.columns) - set(dummies_new.columns)
                for col in missing_cols:
                    dummies_new[col] = 0
                dummies_new = dummies_new[self.X_scaled.columns]  # ordre identique
                # Standardisation
                num_cols = dummies_new.select_dtypes(include=np.number).columns
                dummies_new[num_cols] = self.scaler.transform(dummies_new[num_cols])
                X_new_scaled = dummies_new

            # Ajout constante si nécessaire
            if 'const' not in X_new_scaled.columns:
                X_new_scaled = sm.add_constant(X_new_scaled)

            y_pred = self.model.predict(X_new_scaled)

            if log_transform and self.log_y is not None:
                y_pred = np.exp(y_pred)

            return y_pred
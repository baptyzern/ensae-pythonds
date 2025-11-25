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
class Modelisation :
    def __init__(self , df , lycee_cols , ips , biblio_cols , cible):
        """
        df : DataFrame
        lycee_cols, ips, biblio_cols : listes de colonnes explicatives
        cible : nom de la variable cible
        """
        self.lycee_cols = lycee_cols
        self.ips = ips
        self.biblio_cols = biblio_cols
        
        # Matrice X
        self.X = df[lycee_cols + ips + biblio_cols].copy()
        
        # Variable cible
        self.y = df[cible].copy()
        
        # Ajout de la constante
        self.X = sm.add_constant(self.X)

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

    def Model(self, specify="ols_linear_regression", robust="False", penalization="None", best_lambda=1.0):
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
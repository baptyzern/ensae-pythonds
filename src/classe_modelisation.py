import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Union, Dict, Any
import warnings

class PipelineRegression:
    """
    Pipeline de modélisation de régression linéaire avec visualisations avancées.
    
    Cette classe implémente une approche structurée et reproductible pour la
    modélisation statistique, inspirée des bonnes pratiques en science des données.
    
    Attributs
    ---------
    X : array-like
        Variables explicatives standardisées
    y : array-like
        Variable cible
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Modèle statistique ajusté
    scaler : StandardScaler
        Standardiseur pour les variables explicatives
    standardisation : bool
        Indicateur de standardisation des données
    
    Références (Travaux de Lino Galiana)
    ------------------------------------
    - https://pythonds.linogaliana.fr
    - https://doi.org/10.5281/zenodo.8229676
    """
    
    def __init__(self):
        """Initialise le pipeline de régression."""
        self.X = None
        self.y = None
        self.standardisation = False
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_name = None
        
        # Configuration globale des styles
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'dark': '#2D3047',
            'light': '#C5C5C5'
        }
    
    def data_standardisation(self, data: pd.DataFrame, 
                            numerical_features: List[str]) -> pd.DataFrame:
        """
        Standardise les caractéristiques numériques.
        
        Paramètres
        ----------
        data : pd.DataFrame
            DataFrame contenant les données
        numerical_features : List[str]
            Liste des caractéristiques numériques à standardiser
            
        Retourne
        -------
        pd.DataFrame
            DataFrame avec les caractéristiques standardisées
        """
        if not numerical_features:
            return data
        
        data_standardized = data.copy()
        data_standardized[numerical_features] = self.scaler.fit_transform(
            data[numerical_features]
        )
        return data_standardized
    
    def heatmap_matrix(self, data: pd.DataFrame, 
                       features: List[str], 
                       target: str,
                       figsize: tuple = (10, 8),
                       annot: bool = True) -> plt.Figure:
        """
        Génère une heatmap matricielle de corrélations.
        
        Paramètres
        ----------
        data : pd.DataFrame
            Données d'entrée
        features : List[str]
            Liste des caractéristiques
        target : str
            Variable cible
        standardisation : bool
            Si True, standardise les caractéristiques numériques
        figsize : tuple
            Dimensions de la figure
        annot : bool
            Si True, affiche les valeurs dans les cellules
            
        Retourne
        -------
        plt.Figure
            Figure matplotlib
        """
        # Préparation des données

        data_processed = data.copy()
        
        # Sélection et calcul des corrélations
        numerical_features = [f for f in features if data[f].dtype in ['int64', 'float64']]
        selected_features = numerical_features + [target]
        corr_matrix = data_processed[selected_features].corr()
        
        # Création de la figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Masque pour le triangle supérieur
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Heatmap
        sns.heatmap(corr_matrix,
                   mask=mask,
                   annot=annot,
                   fmt='.2f',
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        # Personnalisation
        ax.set_title('Matrice de Corrélation', 
                    fontsize=16, 
                    fontweight='bold',
                    pad=20)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Rotation des étiquettes
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        return fig
    
    def paires_plot(self, data: pd.DataFrame,
                   features: List[str],
                   target: str,
                   figsize: tuple = (12, 10)) -> plt.Figure:
        """
        Génère un pair plot des caractéristiques avec la variable cible.
        
        Paramètres
        ----------
        data : pd.DataFrame
            Données d'entrée
        features : List[str]
            Liste des caractéristiques
        target : str
            Variable cible
        standardisation : bool
            Si True, standardise les caractéristiques numériques
        figsize : tuple
            Dimensions de la figure
            
        Retourne
        -------
        plt.Figure
            Figure matplotlib
        """
        # Préparation des données

        data_processed = data.copy()
        
        # Limiter le nombre de caractéristiques pour la lisibilité
        if len(features) > 6:
            warnings.warn(f"Le pair plot avec {len(features)} caractéristiques peut être illisible. "
                         "Considérez-en utiliser moins.")
            features = features[:6]
        
        # Création du pair plot
        plot_data = data_processed[features + [target]]
        
        # Création de la figure
        fig = plt.figure(figsize=figsize)
        
        # Création de la grille
        n_features = len(features)
        # Calcul de la hauteur par sous-graphique
        height_per_subplot = figsize[0] / max(n_features, 1)
        
        # Vérifier si la cible est catégorielle pour décider d'utiliser hue
        if plot_data[target].dtype == 'object' or plot_data[target].nunique() < 10:
            g = sns.PairGrid(plot_data, 
                            hue=target,
                            diag_sharey=False,
                            height=height_per_subplot)
        else:
            g = sns.PairGrid(plot_data, 
                            diag_sharey=False,
                            height=height_per_subplot)
        
        # Fonctions pour les différents types de graphiques
        def scatter_func(x, y, **kwargs):
            color = self.colors['primary']
            if 'hue' in kwargs and kwargs['hue'] is not None:
                # Si hue est utilisé, seaborn gère les couleurs
                sns.scatterplot(x=x, y=y, alpha=0.6, s=20, 
                              edgecolor='white', linewidth=0.3, **kwargs)
            else:
                plt.scatter(x, y, alpha=0.6, s=20, 
                          color=color, 
                          edgecolor='white', linewidth=0.3)
                
            if plot_data[target].dtype != 'object' and plot_data[target].nunique() >= 10:
                # Ajout de la ligne de régression pour les cibles continues
                sns.regplot(x=x, y=y, scatter=False, 
                           color=self.colors['secondary'], 
                           line_kws={'linewidth': 1.5})
        
        def hist_func(x, **kwargs):
            plt.hist(x, bins=30, alpha=0.8, 
                    color=self.colors['primary'],
                    edgecolor='white', linewidth=0.5)
            plt.axvline(x.mean(), color=self.colors['secondary'], 
                       linestyle='--', linewidth=2, 
                       label=f'Moyenne: {x.mean():.2f}')
        
        # Application des fonctions
        g.map_upper(scatter_func)
        g.map_lower(scatter_func)
        g.map_diag(hist_func)
        
        # Ajout de la légende pour les histogrammes
        if g.axes[0, 0].get_legend_handles_labels()[0]:
            g.axes[0, 0].legend(loc='best', fontsize=9)
        
        # Titre
        plt.suptitle('Analyse des Relations entre Variables',
                    fontsize=16,
                    fontweight='bold',
                    y=1.02)
        
        # Ajustement de l'espacement
        plt.tight_layout()
        return fig
    
    def fit(self, data: pd.DataFrame,
            features: List[str],
            target: str,
            include_robust: bool = False,
            standardisation: bool = False) -> None:
        """
        Ajuste un modèle de régression linéaire.
        
        Paramètres
        ----------
        data : pd.DataFrame
            Données d'entrée
        features : List[str]
            Liste des caractéristiques
        target : str
            Variable cible
        include_robust : bool
            Si True, utilise les erreurs standards robustes
        standardisation : bool
            Si True, standardise les caractéristiques
        """
        self.standardisation = standardisation
        self.feature_names = features
        self.target_name = target
        
        # Préparation des données
        if standardisation:
            numerical_features = [f for f in features if data[f].dtype in ['int64', 'float64']]
            data_processed = self.data_standardisation(data, numerical_features)
        else:
            data_processed = data.copy()
        
        # Préparation des matrices X et y
        self.X = data_processed[features]
        self.y = data_processed[target]

        # Construction de la formule Patsy à partir des features
        formula = f"{target} ~ {' + '.join(features)}"

        # Ajustement du modèle
        if include_robust:
            self.model = smf.ols(formula=formula, data=data_processed).fit(cov_type="HC0")
        else:
            self.model = smf.ols(formula=formula, data=data_processed).fit()
    
        # Affichage des résultats
        print(self.model.summary())
    
    def predict(self, X_new: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Prédit les valeurs pour de nouvelles observations.
        
        Paramètres
        ----------
        X_new : Union[pd.DataFrame, np.ndarray]
            Nouvelles observations
            
        Retourne
        -------
        np.ndarray
            Prédictions
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été ajusté. Utilisez fit() d'abord.")
        
        # Préparation des données
        if isinstance(X_new, pd.DataFrame):
            # Si c'est un DataFrame, s'assurer qu'il a les bonnes colonnes
            X_new = X_new[self.feature_names]
        else:
            # Si c'est un array numpy, convertir en DataFrame
            X_new = pd.DataFrame(X_new, columns=self.feature_names)
        
        # Standardisation si nécessaire
        if self.standardisation:
            numerical_features = [f for f in self.feature_names 
                                if X_new[f].dtype in ['int64', 'float64']]
            if numerical_features:
                X_new[numerical_features] = self.scaler.transform(X_new[numerical_features])
        
        # Prédiction avec le modèle statsmodels
        # Le modèle OLS de statsmodels.formula.api gère automatiquement les prédictions
        return self.model.predict(X_new)
    
    def plot(self, dimension: int = 2, figsize: tuple = (10, 8)) -> plt.Figure:
        """
        Visualise les résultats de la régression.
        
        Paramètres
        ----------
        dimension : int
            Dimension de la visualisation (2 ou 3)
        figsize : tuple
            Dimensions de la figure
            
        Retourne
        -------
        plt.Figure
            Figure matplotlib
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été ajusté. Utilisez fit() d'abord.")
        
        if self.X.shape[1] == 1 and dimension == 2:
            # Cas 2D : une caractéristique
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Graphique 1 : Nuage de points avec droite de régression
            ax1 = axes[0]
            ax1.scatter(self.X.iloc[:, 0], self.y, alpha=0.6, s=50,
                       color=self.colors['primary'],
                       edgecolor='white', linewidth=0.5,
                       label='Données')
            
            # Droite de régression
            x_range = np.linspace(self.X.iloc[:, 0].min(), self.X.iloc[:, 0].max(), 100)
            x_range_df = pd.DataFrame({self.feature_names[0]: x_range})
            y_pred = self.model.predict(x_range_df)
            
            ax1.plot(x_range, y_pred, 
                    color=self.colors['secondary'],
                    linewidth=2.5,
                    label='Régression')
            
            ax1.set_xlabel(self.feature_names[0], fontsize=12)
            ax1.set_ylabel(self.target_name, fontsize=12)
            ax1.set_title('Régression Linéaire Simple', fontsize=14, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Graphique 2 : Résidus
            ax2 = axes[1]
            residuals = self.model.resid
            ax2.scatter(self.model.fittedvalues, residuals,
                       alpha=0.6, s=50,
                       color=self.colors['accent'],
                       edgecolor='white', linewidth=0.5)
            
            ax2.axhline(y=0, color=self.colors['dark'],
                       linestyle='--', linewidth=1.5)
            ax2.set_xlabel('Valeurs prédites', fontsize=12)
            ax2.set_ylabel('Résidus', fontsize=12)
            ax2.set_title('Analyse des Résidus', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle('Diagnostics de la Régression',
                        fontsize=16,
                        fontweight='bold',
                        y=1.02)
            plt.tight_layout()
            
        else:
            # Cas multivarié : graphique des coefficients
            fig, ax = plt.subplots(figsize=figsize)
            
            # Récupération des coefficients, écart-types et p-values
            # On doit aligner les coefficients avec les feature_names
            coef_dict = {}
            std_err_dict = {}
            pvalues_dict = {}
            
            # Parcourir les paramètres du modèle et extraire les coefficients pour nos features
            for feature in self.feature_names:
                if feature in self.model.params:
                    coef_dict[feature] = self.model.params[feature]
                    std_err_dict[feature] = self.model.bse[feature]
                    pvalues_dict[feature] = self.model.pvalues[feature]
                else:
                    # Si la feature n'est pas trouvée (cas des variables catégorielles transformées)
                    # Chercher les coefficients qui commencent par ce nom
                    matching_coefs = [k for k in self.model.params.index if k.startswith(feature + "[")]
                    if matching_coefs:
                        # Prendre le premier match (attention: cette approche est simplifiée)
                        coef_dict[feature] = self.model.params[matching_coefs[0]]
                        std_err_dict[feature] = self.model.bse[matching_coefs[0]]
                        pvalues_dict[feature] = self.model.pvalues[matching_coefs[0]]
                    else:
                        # Si toujours pas trouvé, mettre à 0
                        coef_dict[feature] = 0
                        std_err_dict[feature] = 0
                        pvalues_dict[feature] = 1
            
            # Créer des listes dans l'ordre des feature_names
            coef = [coef_dict[feature] for feature in self.feature_names]
            std_err = [std_err_dict[feature] for feature in self.feature_names]
            pvalues = [pvalues_dict[feature] for feature in self.feature_names]
            
            # Palette de couleurs pour les p-values (de vert à rouge)
            def get_pvalue_color(pval):
                if pval < 0.001:
                    return '#006400'  # vert très foncé
                elif pval < 0.01:
                    return '#228B22'  # vert forêt
                elif pval < 0.05:
                    return '#32CD32'  # vert lime
                elif pval < 0.1:
                    return '#FFA500'  # orange
                else:
                    return '#DC143C'  # rouge
            
            # Création des couleurs pour chaque coefficient
            colors = [get_pvalue_color(pval) for pval in pvalues]
            
            # Création du bar plot avec des barres très fines
            y_pos = np.arange(len(self.feature_names))
            bar_height = 0.075  # Largeur des Barres
            
            bars = ax.barh(y_pos, coef, 
                        height=bar_height,
                        color=colors,
                        edgecolor='black',
                        linewidth=0.8,
                        alpha=0.9)
            
            # Ajout des valeurs des coefficients avec écart-types entre parenthèses
            for i, (bar, val, err) in enumerate(zip(bars, coef, std_err)):
                # Position du texte dépend du signe du coefficient
                if abs(val) > 0.001:  # Éviter les problèmes avec les très petites valeurs
                    x_pos = val + (0.01 if val >= 0 else -0.01)
                else:
                    x_pos = 0.01  # Valeur par défaut pour les coefficients proches de 0
                
                ha = 'left' if val >= 0 else 'right'
                
                # Texte : coefficient (écart-type)
                text = f'{val:.3f} ({err:.3f})'
                
                # Couleur du texte : noir pour meilleure lisibilité
                text_color = 'black'
                
                # Ajout du texte
                ax.text(x_pos, 
                    bar.get_y() + bar.get_height()/2,
                    text,
                    va='center',
                    ha=ha,
                    fontsize=9,
                    fontweight='medium',
                    color=text_color,
                    bbox=dict(boxstyle='round,pad=0.2', 
                                facecolor='white', 
                                alpha=0.85,
                                edgecolor='lightgray',
                                linewidth=0.5))
            
            # Configuration de l'axe Y
            ax.set_yticks(y_pos)
            ax.set_yticklabels(self.feature_names, fontsize=11, fontweight='medium')
            
            # Configuration de l'axe X
            ax.set_xlabel('Valeur du coefficient', fontsize=12)
            ax.set_title('Coefficients de la régression', 
                        fontsize=14, 
                        fontweight='bold',
                        pad=20)
            
            # Ligne verticale à zéro
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
            
            # Grille uniquement sur l'axe X
            ax.grid(True, alpha=0.2, axis='x', linestyle='--')
            
            # Création d'une barre de couleur (colorbar) pour les p-values
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize, ListedColormap
            
            # Définition des couleurs et des étiquettes pour la colorbar
            colorbar_colors = ['#006400', '#228B22', '#32CD32', '#FFA500', '#DC143C']
            colorbar_labels = ['p < 0.001', 'p < 0.01', 'p < 0.05', 'p < 0.1', 'p ≥ 0.1']
            colorbar_bounds = [0, 0.001, 0.01, 0.05, 0.1, 1.0]
            
            # Création d'une colormap discrète
            cmap = ListedColormap(colorbar_colors)
            
            # Création d'un mappable pour la colorbar
            norm = Normalize(vmin=0, vmax=1)
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            # Ajout de la colorbar à droite du graphique
            cbar = fig.colorbar(sm, ax=ax, orientation='vertical', 
                            pad=0.02, shrink=0.8, aspect=20)
            
            # Configuration des ticks de la colorbar
            tick_positions = [0.1, 0.3, 0.5, 0.7, 0.9]  # Positions des ticks au centre de chaque couleur
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(colorbar_labels, fontsize=8)
            
            # Ajustement des limites pour s'assurer que tout est visible
            x_min, x_max = ax.get_xlim()
            x_range = x_max - x_min
            ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
            
            # Ajustement de la disposition pour faire de la place à la colorbar
            plt.subplots_adjust(right=0.85)
            
            plt.tight_layout()
        return fig
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Union, Dict, Any, Tuple
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.gridspec as gridspec

class PipelineRegression:
    """
    Pipeline de modélisation de régression linéaire.
    
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
    stepwise_best : statsmodels.regression.linear_model.RegressionResultsWrapper
        Meilleur modèle de la sélection stepwise
        
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
        self.stepwise_best = None
        
        # Configuration globale des styles
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'dark': '#2D3047',
            'light': '#C5C5C5',
            'warning': '#FF6B6B',
            'safe': '#4ECDC4'
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
                       annot: bool = True,
                       ax: Optional[plt.Axes] = None) -> plt.Figure:
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
        figsize : tuple
            Dimensions de la figure
        annot : bool
            Si True, affiche les valeurs dans les cellules
        ax : plt.Axes, optional
            Axe matplotlib sur lequel dessiner
            
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
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
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
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        if ax is None:
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
                sns.scatterplot(x=x, y=y, alpha=0.6, s=20, 
                              edgecolor='white', linewidth=0.3, **kwargs)
            else:
                plt.scatter(x, y, alpha=0.6, s=20, 
                          color=color, 
                          edgecolor='white', linewidth=0.3)
                
            if plot_data[target].dtype != 'object' and plot_data[target].nunique() >= 10:
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
    
    def vif_plots(self, data: pd.DataFrame,
                 features: List[str],
                 target: str = None,
                 figsize: tuple = (12, 6),
                 ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Calcule et visualise les VIF (Variance Inflation Factor) des variables.
        
        Paramètres
        ----------
        data : pd.DataFrame
            Données d'entrée
        features : List[str]
            Liste des caractéristiques
        target : str, optional
            Variable cible (non utilisée dans le VIF mais pour la cohérence)
        figsize : tuple
            Dimensions de la figure
        ax : plt.Axes, optional
            Axe matplotlib sur lequel dessiner
            
        Retourne
        -------
        plt.Figure
            Figure matplotlib
        """
        # Préparation des données
        data_processed = data[features].copy()
        
        # Vérifier que toutes les variables sont numériques
        non_numeric = [f for f in features if data_processed[f].dtype not in ['int64', 'float64']]
        if non_numeric:
            warnings.warn(f"Les variables suivantes ne sont pas numériques et seront exclues: {non_numeric}")
            features = [f for f in features if f not in non_numeric]
            data_processed = data_processed[features]
        
        # Ajouter une constante pour le calcul du VIF
        data_processed = pd.DataFrame(data_processed).assign(const=1)
        
        # Calculer le VIF pour chaque variable
        vif_data = pd.DataFrame()
        vif_data["Variable"] = features
        vif_data["VIF"] = [variance_inflation_factor(data_processed.values, i) 
                          for i in range(len(features))]
        
        # Tri par VIF décroissant
        vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)
        
        # Création de la figure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Création du graphique à barres
        bars = ax.bar(vif_data["Variable"], vif_data["VIF"], 
                      width=0.25,  
                      color=[self.colors['warning'] if v > 10 else self.colors['primary'] 
                             for v in vif_data["VIF"]])
        
        # Ligne de seuil VIF = 10
        ax.axhline(y=10, color=self.colors['warning'], linestyle='--', 
                  linewidth=2, label='VIF = 10')
        
        # Ligne de seuil VIF = 5
        ax.axhline(y=5, color=self.colors['accent'], linestyle=':', 
                  linewidth=1.5, label='VIF = 5')
        
        # Personnalisation
        ax.set_ylabel('VIF', fontsize=12, fontweight='bold')
        ax.set_title('Diagnostic de Multicolinéarité', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Rotation des étiquettes des variables
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Légende
        ax.legend(loc='upper right', fontsize=10)
        
        # Grille
        ax.grid(True, alpha=0.3, axis='y')
        
        # Ajustement des limites
        max_vif = max(vif_data["VIF"])
        ax.set_ylim(0, max(max_vif * 1.1, 12))
        
        # Informations textuelles
        high_vif = vif_data[vif_data["VIF"] > 10]
        moderate_vif = vif_data[(vif_data["VIF"] > 5) & (vif_data["VIF"] <= 10)]
        
        info_text = []
        if not high_vif.empty:
            info_text.append(f"Multicolinéarité élevée (VIF > 10): {len(high_vif)} variable(s)")
        if not moderate_vif.empty:
            info_text.append(f"Multicolinéarité modérée (5 < VIF ≤ 10): {len(moderate_vif)} variable(s)")
        
        if info_text:
            info_str = "\n".join(info_text)
            ax.text(0.02, 0.98, info_str, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if ax is None:
            plt.tight_layout()
        
        return fig
    
    def headmap_vif(self, data: pd.DataFrame,
                               features: List[str],
                               target: str,
                               figsize: tuple = (24, 8)) -> plt.Figure:
        """
        Crée un dashboard complet avec 3 visualisations principales.
        
        Paramètres
        ----------
        data : pd.DataFrame
            Données d'entrée
        features : List[str]
            Liste des caractéristiques
        target : str
            Variable cible
        figsize : tuple
            Dimensions de la figure
            
        Retourne
        -------
        plt.Figure
            Figure matplotlib avec les 3 graphiques
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Heatmap
        self.heatmap_matrix(data=data, features=features, target=target, ax=axes[0])
        axes[0].set_title('Matrice de Corrélation', fontsize=14, fontweight='bold')
        
        # 2. VIF plots
        self.vif_plots(data=data, features=features, target=target, ax=axes[1])
        axes[1].set_title('Diagnostic de Multicolinéarité (VIF)', fontsize=14, fontweight='bold')
        
        
        # Ajustement de l'espacement
        plt.tight_layout()
        return fig

    def fit_stepwise(self, data: pd.DataFrame,
                 features: List[str],
                 target: str,
                 include_robust: bool = False,
                 standardisation: bool = False,
                 forward: bool = True,
                 verbose: bool = True,
                 best : bool = False) -> None:
        """
        Implémente une sélection stepwise des variables (forward ou backward) 
        basée uniquement sur l'AIC avec affichage des tableaux statsmodels complets.
        
        Paramètres
        ----------
        data : pd.DataFrame
            Données d'entrée
        features : List[str]
            Liste des caractéristiques initiales
        target : str
            Variable cible
        include_robust : bool
            Si True, utilise les erreurs standards robustes
        standardisation : bool
            Si True, standardise les caractéristiques
        forward : bool
            Si True, utilise forward selection, sinon backward elimination
        verbose : bool
            Si True, affiche le tableau model.summary() uniquement pour le modèle retenu à chaque étape
        best : bool
            Si True, affiche uniquement le meilleur modèle final
            
        Retourne
        -------
        None
            Stocke le meilleur modèle dans self.stepwise_best
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
        
        if forward:
            self._forward_selection_aic(data_processed, features, target, include_robust, verbose)
            if best :
                print("="*60)
                print("MEILLEUR MODÈLE AVEC FORWARD SELECTION")
                print("="*60)
                print(self.stepwise_best.summary())
        else:
            self._backward_elimination_aic(data_processed, features, target, include_robust, verbose)
            if best :
                print("="*60)
                print("MEILLEUR MODÈLE AVEC BACKWARD SELECTION")
                print("="*60)
                print(self.stepwise_best.summary())
                

    def _forward_selection_aic(self, data: pd.DataFrame,
                            features: List[str],
                            target: str,
                            include_robust: bool,
                            verbose: bool) -> None:
        """
        Implémente la sélection forward basée uniquement sur l'AIC.
        Affiche uniquement le tableau du modèle retenu à chaque étape.
        """
        selected_features = []
        remaining_features = features.copy()
        iteration = 0
        
        while remaining_features:
            iteration += 1
            best_candidate = None
            best_aic = float('inf')
            best_model = None
            
            # Tester chaque variable candidate (sans afficher les tableaux)
            for candidate in remaining_features:
                test_features = selected_features + [candidate]
                formula = f"{target} ~ {' + '.join(test_features)}"
                
                try:
                    if include_robust:
                        model = smf.ols(formula=formula, data=data).fit(cov_type="HC0")
                    else:
                        model = smf.ols(formula=formula, data=data).fit()
                    
                    # Mettre à jour la meilleure candidate basée sur AIC
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_candidate = candidate
                        best_model = model
                        
                except Exception as e:
                    print(f"  - {candidate}: Erreur - {str(e)}")
            
            # Vérifier s'il y a une amélioration
            if iteration == 1 or best_aic < self.stepwise_best.aic:
                selected_features.append(best_candidate)
                remaining_features.remove(best_candidate)
                self.stepwise_best = best_model
                
                # Afficher UNIQUEMENT le tableau du modèle retenu (si verbose=True)
                if verbose:
                    print("\n")
                    print(f"================ÉTAPE {iteration} - FORWARD SELECTION================= \n")
                    print(self.stepwise_best.summary())
            else:
                break

    def _backward_elimination_aic(self, data: pd.DataFrame,
                                features: List[str],
                                target: str,
                                include_robust: bool,
                                verbose: bool) -> None:
        """
        Implémente l'élimination backward basée uniquement sur l'AIC.
        Affiche uniquement le tableau du modèle retenu à chaque étape.
        """
        current_features = features.copy()
        iteration = 0
        
        # Modèle initial avec toutes les variables
        formula = f"{target} ~ {' + '.join(current_features)}"
        if include_robust:
            current_model = smf.ols(formula=formula, data=data).fit(cov_type="HC0")
        else:
            current_model = smf.ols(formula=formula, data=data).fit()
        
        self.stepwise_best = current_model
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"MODÈLE INITIAL")
            print(f"{'='*60}")
            print(self.stepwise_best.summary())
        
        while len(current_features) > 1:
            iteration += 1
            best_aic = float('inf')
            best_features = None
            best_model = None
            removed_feature = None
            
            
            # Tester chaque suppression possible (sans afficher les tableaux)
            for feature_to_remove in current_features:
                test_features = [f for f in current_features if f != feature_to_remove]
                formula = f"{target} ~ {' + '.join(test_features)}"
                
                try:
                    if include_robust:
                        model = smf.ols(formula=formula, data=data).fit(cov_type="HC0")
                    else:
                        model = smf.ols(formula=formula, data=data).fit()
                    
                    # Trouver le meilleur modèle (AIC le plus bas)
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_features = test_features
                        best_model = model
                        removed_feature = feature_to_remove
                        
                except Exception as e:
                    print(f"  - Sans {feature_to_remove}: Erreur - {str(e)}")
            
            # Vérifier si la suppression améliore l'AIC
            if best_aic < self.stepwise_best.aic:
                current_features = best_features
                self.stepwise_best = best_model
                
                # Afficher UNIQUEMENT le tableau du modèle retenu (si verbose=True)
                if verbose:
                    print("\n")
                    print(f"================ÉTAPE {iteration} - BACKWARD SELECTION=================\n")
                    print(self.stepwise_best.summary())
            else:
                break



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
        
        # Construction de la formule Patsy
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
            X_new = X_new[self.feature_names]
        else:
            X_new = pd.DataFrame(X_new, columns=self.feature_names)
        
        # Standardisation si nécessaire
        if self.standardisation:
            numerical_features = [f for f in self.feature_names 
                                if X_new[f].dtype in ['int64', 'float64']]
            if numerical_features:
                X_new[numerical_features] = self.scaler.transform(X_new[numerical_features])
        
        # Prédiction
        return self.model.predict(X_new)
    
    def diagnostic_plots(self, data: pd.DataFrame,
                        figsize: tuple = (15, 10)) -> plt.Figure:
        """
        Crée des graphiques de diagnostic pour le modèle ajusté.
        
        Paramètres
        ----------
        data : pd.DataFrame
            Données d'entrée
        figsize : tuple
            Dimensions de la figure
            
        Retourne
        -------
        plt.Figure
            Figure matplotlib avec les diagnostics
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été ajusté. Utilisez fit() d'abord.")
        
        # Prédictions et résidus
        y_pred = self.predict(data[self.feature_names])
        residuals = data[self.target_name] - y_pred
        standardized_residuals = residuals / np.std(residuals)
        
        # Création de la figure
        fig = plt.figure(figsize=figsize)
        
        # Grille 2x2 pour les diagnostics
        gs = gridspec.GridSpec(2, 2)
        
        # 1. Résidus vs Prédictions
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(y_pred, residuals, alpha=0.6, color=self.colors['primary'])
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel('Prédictions', fontsize=12)
        ax1.set_ylabel('Résidus', fontsize=12)
        ax1.set_title('Résidus vs Prédictions', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. QQ-plot des résidus
        ax2 = fig.add_subplot(gs[0, 1])
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('QQ-plot des Résidus', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Histogramme des résidus
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(residuals, bins=30, alpha=0.7, color=self.colors['primary'],
                edgecolor='white', linewidth=0.5)
        ax3.axvline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Résidus', fontsize=12)
        ax3.set_ylabel('Fréquence', fontsize=12)
        ax3.set_title('Distribution des Résidus', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Résidus standardisés
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.scatter(range(len(standardized_residuals)), standardized_residuals,
                   alpha=0.6, color=self.colors['primary'])
        ax4.axhline(y=0, color='red', linestyle='-', linewidth=1)
        ax4.axhline(y=2, color=self.colors['warning'], linestyle='--', linewidth=1)
        ax4.axhline(y=-2, color=self.colors['warning'], linestyle='--', linewidth=1)
        ax4.set_xlabel('Index des Observations', fontsize=12)
        ax4.set_ylabel('Résidus Standardisés', fontsize=12)
        ax4.set_title('Résidus Standardisés', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Informations statistiques
        info_text = (f"R²: {self.model.rsquared:.3f}\n"
                    f"R² ajusté: {self.model.rsquared_adj:.3f}\n"
                    f"MSE: {np.mean(residuals**2):.3f}\n"
                    f"Test de normalité (Shapiro-Wilk):\n"
                    f"  p-value: {stats.shapiro(residuals)[1]:.4f}")
        
        fig.text(0.02, 0.98, info_text, transform=fig.transFigure,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
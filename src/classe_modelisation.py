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
from scipy import stats
import statsmodels.api as sm
from scipy.stats import chi2_contingency, pointbiserialr
import itertools

class PipelineRegression:
    """
    Pipeline de modélisation de régression linéaire avec gestion étendue des variables catégorielles.
    
    Cette classe implémente une approche structurée pour l'analyse exploratoire,
    le diagnostic et la modélisation statistique, adaptée aux jeux de données
    avec des variables mixtes (catégorielles et continues).
    
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
    feature_names : List[str]
        Noms des caractéristiques utilisées
    target_name : str
        Nom de la variable cible
    cat_features : List[str]
        Liste des variables catégorielles
    num_features : List[str]
        Liste des variables numériques
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
        self.cat_features = []
        self.num_features = []
        
        # Configuration globale des styles
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Palette de couleurs professionnelle
        self.colors = {
            'primary': '#2E86AB',      # Bleu professionnel
            'secondary': '#A23B72',    # Violet
            'accent': '#F18F01',       # Orange
            'dark': '#2D3047',         # Noir bleuté
            'light': '#C5C5C5',        # Gris clair
            'warning': '#FF6B6B',      # Rouge d'avertissement
            'safe': '#4ECDC4',         # Vert turquoise
            'neutral': '#7B8B8B'       # Gris neutre
        }
    
    # ============================================================================
    # MÉTHODES INTERNES
    # ============================================================================
    
    def _identify_feature_types(self, data: pd.DataFrame, features: List[str]) -> None:
        """
        Identifie automatiquement les types de variables (catégorielles vs numériques).
        
        Paramètres
        ----------
        data : pd.DataFrame
            DataFrame contenant les données
        features : List[str]
            Liste des caractéristiques à analyser
        """
        self.cat_features = []
        self.num_features = []
        
        for feature in features:
            # Critères pour identifier une variable catégorielle
            is_categorical = (
                data[feature].dtype == 'object' or 
                data[feature].dtype.name == 'category' or
                data[feature].nunique() < 10 or
                (data[feature].dtype == 'int64' and data[feature].nunique() < 10)
            )
            
            if is_categorical:
                self.cat_features.append(feature)
            elif data[feature].dtype in ['int64', 'float64']:
                self.num_features.append(feature)
    
    def _safe_correlation(self, data: pd.DataFrame, var1: str, var2: str) -> float:
        """
        Calcule la corrélation de manière robuste en gérant les NaN.
        """
        clean_data = data[[var1, var2]].dropna()
        if len(clean_data) < 2:
            return 0.0
        return clean_data.corr().iloc[0, 1]
    
    # ============================================================================
    # ANALYSE EXPLORATOIRE VISUELLE
    # ============================================================================
    
    def boxplots_categoriels(self, data: pd.DataFrame,
                            features: List[str],
                            target: str,
                            figsize: tuple = (15, 10),
                            n_cols: int = 3,
                            max_categories: int = 20) -> plt.Figure:
        """
        Génère des boxplots de la variable cible par catégorie avec sous-figures.
        
        Paramètres
        ----------
        data : pd.DataFrame
            Données d'entrée
        features : List[str]
            Liste des caractéristiques à analyser
        target : str
            Variable cible continue
        figsize : tuple, optional
            Dimensions de la figure, par défaut (15, 10)
        n_cols : int, optional
            Nombre de colonnes pour l'agencement des sous-figures, par défaut 3
        max_categories : int, optional
            Nombre maximum de catégories à afficher par variable, par défaut 20
            
        Retourne
        -------
        plt.Figure
            Figure matplotlib avec les boxplots
        """
        # Identification des variables catégorielles
        self._identify_feature_types(data, features)
        cat_features_to_plot = self.cat_features
        
        if not cat_features_to_plot:
            warnings.warn("Aucune variable catégorielle détectée pour les boxplots.")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Aucune variable catégorielle detectee",
                   ha='center', va='center', fontsize=12)
            return fig
        
        n_features = len(cat_features_to_plot)
        n_rows = int(np.ceil(n_features / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Convertir axes en liste pour manipulation facile
        if n_features > 1:
            axes = axes.flatten()
        elif n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.ravel()
        
        for idx, feature in enumerate(cat_features_to_plot):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Filtrer les données sans NaN pour la catégorie et la target
            plot_data = data[[feature, target]].dropna()
            
            if plot_data.empty or len(plot_data) < 5:
                ax.text(0.5, 0.5, "Donnees insuffisantes",
                       ha='center', va='center', fontsize=10)
                ax.set_title(f'{feature} (n/a)', fontsize=11)
                continue
            
            # Calculer les médianes et créer un ordre unique
            try:
                median_series = plot_data.groupby(feature)[target].median()
                
                # Si trop de catégories, on garde les principales
                if len(median_series) > max_categories:
                    # Garder les catégories les plus fréquentes
                    counts = plot_data[feature].value_counts()
                    top_categories = counts.nlargest(max_categories).index.tolist()
                    plot_data = plot_data[plot_data[feature].isin(top_categories)]
                    median_series = plot_data.groupby(feature)[target].median()
                    
                # Créer un DataFrame avec médiane et catégorie
                median_df = median_series.reset_index()
                median_df.columns = ['categorie', 'mediane']
                
                # Trier par médiane, puis par catégorie pour un ordre déterministe
                median_df = median_df.sort_values(['mediane', 'categorie'], ascending=False)
                order = median_df['categorie'].tolist()
                
                # Boxplot
                boxplot = sns.boxplot(x=feature, y=target, data=plot_data,
                                     order=order,
                                     ax=ax,
                                     palette='Set2',
                                     showmeans=True,
                                     meanprops={"marker": "D", 
                                               "markerfacecolor": "white", 
                                               "markeredgecolor": self.colors['warning'],
                                               "markersize": 5})
                
                # Ajout des moyennes si nombre raisonnable de catégories
                if len(order) <= 15:
                    means = plot_data.groupby(feature)[target].mean()
                    for i, cat in enumerate(order):
                        if cat in means.index:
                            mean_val = means[cat]
                            # Positionner le texte au-dessus de la boîte
                            ax.text(i, mean_val, f'{mean_val:.1f}', 
                                   ha='center', va='bottom', fontsize=8, 
                                   color=self.colors['dark'])
                
                # Personnalisation
                ax.set_title(f'{feature}\n({len(order)} categories)', fontsize=11, pad=10)
                ax.set_xlabel('')
                ax.set_ylabel(target if idx % n_cols == 0 else '')
                
                # Ajuster la rotation selon le nombre de catégories
                rotation = 45 if len(order) > 5 else (90 if len(order) > 10 else 0)
                ax.tick_params(axis='x', rotation=rotation, labelsize=9)
                
                # Grille subtile
                ax.grid(True, alpha=0.2, axis='y', linestyle='--')
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Erreur: {str(e)[:30]}",
                       ha='center', va='center', fontsize=9)
                ax.set_title(f'{feature} (erreur)', fontsize=11)
                continue
        
        # Masquer les axes inutilisés
        for idx in range(len(cat_features_to_plot), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Distribution de la Variable Cible par Categorie',
                    fontsize=16, fontweight='bold', y=1.02)
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
    
    # ============================================================================
    # ANALYSE D'ASSOCIATION
    # ============================================================================
    
    def mixed_association_heatmap(self, data: pd.DataFrame,
                                 features: List[str],
                                 target: Optional[str] = None,
                                 figsize: tuple = (12, 10)) -> plt.Figure:
        """
        Heatmap d'association hétérogène (mélange de variables continues et catégorielles).
        
        Utilise :
        - Pearson pour continue vs continue
        - Corrélation bisériale ponctuelle pour continue vs binaire
        - Rapport de corrélation (eta) pour continue vs catégorielle (>2 modalités)
        - V de Cramér pour catégorielle vs catégorielle
        
        Paramètres
        ----------
        data : pd.DataFrame
            Données d'entrée
        features : List[str]
            Liste des caractéristiques
        target : str, optional
            Variable cible (si None, matrice symétrique)
        figsize : tuple, optional
            Dimensions de la figure, par défaut (12, 10)
            
        Retourne
        -------
        plt.Figure
            Figure matplotlib
        """
        # Identifier les types de variables
        self._identify_feature_types(data, features)
        
        all_vars = features.copy()
        if target and target not in all_vars:
            all_vars.append(target)
        
        # Créer une matrice de similarité
        n_vars = len(all_vars)
        assoc_matrix = pd.DataFrame(np.zeros((n_vars, n_vars)),
                                   index=all_vars,
                                   columns=all_vars)
        
        # Remplir la diagonale avec 1
        np.fill_diagonal(assoc_matrix.values, 1.0)
        
        # Remplir la matrice d'association
        for i, var1 in enumerate(all_vars):
            for j, var2 in enumerate(all_vars):
                if i >= j:  # Matrice symétrique, on calcule seulement la moitié
                    continue
                
                # Déterminer les types
                type_var1 = 'cat' if var1 in self.cat_features else 'num'
                type_var2 = 'cat' if var2 in self.cat_features else 'num'
                
                # Sélectionner les données valides
                valid_data = data[[var1, var2]].dropna()
                if len(valid_data) < 2:
                    continue
                
                # Selon la combinaison de types
                if type_var1 == 'num' and type_var2 == 'num':
                    # Pearson
                    corr = valid_data[var1].corr(valid_data[var2])
                    assoc_value = abs(corr) if not np.isnan(corr) else 0
                    
                elif type_var1 == 'cat' and type_var2 == 'cat':
                    # V de Cramér
                    contingency_table = pd.crosstab(valid_data[var1], valid_data[var2])
                    try:
                        chi2, p, dof, expected = chi2_contingency(contingency_table)
                        n_obs = contingency_table.sum().sum()
                        cramers_v = np.sqrt(chi2 / (n_obs * (min(contingency_table.shape) - 1)))
                        assoc_value = cramers_v if not np.isnan(cramers_v) else 0
                    except:
                        assoc_value = 0
                    
                else:
                    # Mixte: déterminer quelle variable est catégorielle
                    cat_var = var1 if type_var1 == 'cat' else var2
                    num_var = var2 if type_var1 == 'cat' else var1
                    
                    n_categories = valid_data[cat_var].nunique()
                    
                    if n_categories == 2:
                        # Corrélation bisériale ponctuelle
                        categories = valid_data[cat_var].unique()
                        if len(categories) == 2:
                            cat_encoded = valid_data[cat_var].map({categories[0]: 0, categories[1]: 1})
                            try:
                                corr, p_value = pointbiserialr(cat_encoded, valid_data[num_var])
                                assoc_value = abs(corr) if not np.isnan(corr) else 0
                            except:
                                assoc_value = 0
                        else:
                            assoc_value = 0
                    else:
                        # Rapport de corrélation (eta carré)
                        groups = [valid_data[valid_data[cat_var] == cat][num_var] 
                                 for cat in valid_data[cat_var].unique()]
                        
                        # Calculer la variance expliquée
                        overall_mean = valid_data[num_var].mean()
                        ss_between = sum([len(g) * (g.mean() - overall_mean)**2 for g in groups])
                        ss_total = sum((valid_data[num_var] - overall_mean)**2)
                        
                        eta_squared = ss_between / ss_total if ss_total > 0 else 0
                        assoc_value = np.sqrt(eta_squared)
                
                # Remplir la matrice (symétrique)
                assoc_matrix.iloc[i, j] = assoc_value
                assoc_matrix.iloc[j, i] = assoc_value
        
        # Création de la heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        # Masque pour le triangle supérieur si pas de cible spécifique
        if not target:
            mask = np.triu(np.ones_like(assoc_matrix, dtype=bool))
        else:
            mask = None
        
        # Heatmap avec annotations
        sns.heatmap(assoc_matrix,
                   mask=mask,
                   annot=True,
                   fmt='.2f',
                   cmap='YlOrRd',
                   vmin=0,
                   vmax=1,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8, "label": "Intensite d'association"},
                   ax=ax)
        
        # Personnalisation
        if target:
            title = f"Matrice d'Association Heterogene (avec cible : {target})"
        else:
            title = "Matrice d'Association Heterogene"
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Ajouter des indications sur les types de variables
        cat_vars = [v for v in all_vars if v in self.cat_features]
        num_vars = [v for v in all_vars if v not in self.cat_features]
        
        # Légende pour les types de variables
        legend_text = []
        if cat_vars:
            legend_text.append("Variables categorielles :")
            legend_text.extend([f"  - {var}" for var in cat_vars])
        if num_vars:
            legend_text.append("\nVariables numeriques :")
            legend_text.extend([f"  - {var}" for var in num_vars])
        
        if legend_text:
            ax.text(1.02, 0.95, '\n'.join(legend_text), transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        return fig
    
    # ============================================================================
    # MÉTHODES DE MODÉLISATION
    # ============================================================================
    
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
        include_robust : bool, optional
            Si True, utilise les erreurs standards robustes, par défaut False
        standardisation : bool, optional
            Si True, standardise les caractéristiques numériques, par défaut False
        """
        self.standardisation = standardisation
        self.feature_names = features
        self.target_name = target
        
        # Identifier les types
        self._identify_feature_types(data, features)
        
        # Préparation des données
        if standardisation:
            # Standardiser uniquement les variables numériques
            data_processed = data.copy()
            if self.num_features:
                data_processed[self.num_features] = self.scaler.fit_transform(
                    data[self.num_features]
                )
        else:
            data_processed = data.copy()
        
        # Construction de la formule
        formula = f"{target} ~ {' + '.join(features)}"
        
        # Ajustement du modèle
        try:
            if include_robust:
                self.model = smf.ols(formula=formula, data=data_processed).fit(cov_type="HC0")
            else:
                self.model = smf.ols(formula=formula, data=data_processed).fit()
            
            print("=" * 80)
            print("RESULTATS DE LA REGRESSION LINEAIRE")
            print("=" * 80)
            print(self.model.summary())
            
        except Exception as e:
            print(f"Erreur lors de l'ajustement du modele : {str(e)}")
            self.model = None
    



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
            
        Lève
        -----
        ValueError
            Si le modèle n'a pas été ajusté
        """
        if self.model is None:
            raise ValueError("Le modele n'a pas ete ajuste. Utilisez fit() d'abord.")
        
        # Préparation des données
        if isinstance(X_new, pd.DataFrame):
            X_new = X_new[self.feature_names]
        else:
            X_new = pd.DataFrame(X_new, columns=self.feature_names)
        
        # Standardisation si nécessaire
        if self.standardisation and self.num_features:
            numerical_features = [f for f in self.feature_names if f in self.num_features]
            if numerical_features:
                X_new[numerical_features] = self.scaler.transform(X_new[numerical_features])
        
        # Prédiction
        return self.model.predict(X_new)
    
    # ============================================================================
    # MÉTHODES UTILITAIRES
    # ============================================================================
    
    def summary_statistics(self, data: pd.DataFrame,
                          features: List[str],
                          target: str) -> pd.DataFrame:
        """
        Génère un résumé statistique des variables.
        
        Paramètres
        ----------
        data : pd.DataFrame
            Données d'entrée
        features : List[str]
            Liste des caractéristiques
        target : str
            Variable cible
            
        Retourne
        -------
        pd.DataFrame
            Résumé statistique
        """
        self._identify_feature_types(data, features)
        
        summary_list = []
        
        # Variables numériques
        for var in self.num_features:
            var_data = data[var].dropna()
            summary_list.append({
                'Variable': var,
                'Type': 'Numerique',
                'N': len(var_data),
                'Moyenne': var_data.mean(),
                'Ecart-type': var_data.std(),
                'Min': var_data.min(),
                'Q1': var_data.quantile(0.25),
                'Mediane': var_data.median(),
                'Q3': var_data.quantile(0.75),
                'Max': var_data.max(),
                'NA': data[var].isna().sum(),
                '% NA': data[var].isna().mean() * 100
            })
        
        # Variables catégorielles
        for var in self.cat_features:
            var_data = data[var].dropna()
            unique_vals = var_data.nunique()
            top_cat = var_data.value_counts().index[0] if not var_data.empty else None
            top_freq = var_data.value_counts().iloc[0] if not var_data.empty else 0
            
            summary_list.append({
                'Variable': var,
                'Type': 'Categorielle',
                'N': len(var_data),
                'Modalites': unique_vals,
                'Modalite_freq': top_cat,
                'Freq_max': top_freq,
                '% Freq_max': (top_freq / len(var_data) * 100) if len(var_data) > 0 else 0,
                'Min': np.nan,
                'Q1': np.nan,
                'Mediane': np.nan,
                'Q3': np.nan,
                'Max': np.nan,
                'NA': data[var].isna().sum(),
                '% NA': data[var].isna().mean() * 100
            })
        
        summary_df = pd.DataFrame(summary_list)
        
        print("=" * 80)
        print("RESUME STATISTIQUE DES VARIABLES")
        print("=" * 80)
        print(f"\nVariables numeriques : {len(self.num_features)}")
        print(f"Variables categorielles : {len(self.cat_features)}")
        print(f"Variable cible : {target}")
        print("\n" + "-" * 80)
        print(summary_df.to_string(index=False))
        print("-" * 80)
        
        return summary_df
    
    def export_results(self, output_dir: str = "./results") -> None:
        """
        Exporte les résultats et graphiques.
        
        Paramètres
        ----------
        output_dir : str, optional
            Répertoire de sortie, par défaut "./results"
        """
        import os
        
        # Créer le répertoire si nécessaire
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Export des resultats dans : {output_dir}")
        
        # Exporter les paramètres du modèle si disponible
        if self.model is not None:
            params_df = pd.DataFrame({
                'Variable': self.model.params.index,
                'Coefficient': self.model.params.values,
                'Std_Error': self.model.bse.values,
                't_value': self.model.tvalues.values,
                'p_value': self.model.pvalues.values
            })
            params_df.to_csv(f"{output_dir}/coefficients.csv", index=False)
            print(f"  - Coefficients exportes : {output_dir}/coefficients.csv")
        
        print("Export termine.")
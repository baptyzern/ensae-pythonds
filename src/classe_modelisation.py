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
from IPython.display import display, HTML

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

        mask = np.triu(np.ones_like(assoc_matrix, dtype=bool))
        
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
        title = "Matrice d'Association Heterogene"
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
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
            standardisation: bool = False,
            first: Optional[bool] = None,
            rayon: Optional[int] = None) -> None:
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
        if first is not None:
            if first:
                text = "Tableau 1 : Impact de la présence/absence de bibliothèques"
            else:
                text = "Tableau 2 : Impact du nombre de bibliothèques"
        else:
            text = "Tableau : Résultats de la régression linéaire"

        if rayon is not None:
            text += f" dans un rayon de {rayon} mètres"
                
        try:
            if include_robust:
                self.model = smf.ols(formula=formula, data=data_processed).fit(cov_type="HC0")
            else:
                self.model = smf.ols(formula=formula, data=data_processed).fit()
            html_summary = self.model.summary().as_html()
            display(HTML(f"""
                <h3 style="color:darkgreen; font-weight:bold;">{text}</h3>
                <div style="color:darkblue;font-weight:bold; font-family:Arial;">
                    {html_summary}
                </div>
            """))
            
        except Exception as e:
            print(f"Erreur lors de l'ajustement du modele : {str(e)}")
            self.model = None

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
    def plot_coefficients(self):
        # Configuration pour un style professionnel
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

        # Données des coefficients
        rayons = np.array([0.5, 1.0, 2.0, 5.0])  # en km

        # Effet de présence (dummy)
        coef_presence = np.array([0.9627, 1.2921, 1.2010, 1.3437])
        std_presence = np.array([0.401, 0.386, 0.539, 0.696])
        conf_presence = 1.96 * std_presence  # Intervalle de confiance 95%

        # Effet de densité (nombre)
        coef_densite = np.array([0.4104, 0.1297, 0.0415, 0.0154])
        std_densite = np.array([0.193, 0.075, 0.035, 0.009])
        conf_densite = 1.96 * std_densite

        # Création de la figure avec deux sous-graphiques
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

        # Graphique 1 : Effet de présence (convexe)
        ax1.errorbar(rayons, coef_presence, yerr=conf_presence, 
                    fmt='o', color='#1f77b4', capsize=6, capthick=1.5, 
                    linewidth=1.5, markersize=8, label='Estimation ponctuelle')

        # Courbe de tendance polynomiale (degré 2 pour illustrer la convexité)
        poly_coeff = np.polyfit(rayons, coef_presence, 2)
        poly_fit = np.poly1d(poly_coeff)
        rayons_smooth = np.linspace(0.3, 5.5, 100)
        ax1.plot(rayons_smooth, poly_fit(rayons_smooth), '--', color='#1f77b4', 
                alpha=0.7, linewidth=1.5, label='Tendance convexe')

        ax1.set_xlabel('Rayon (km)', fontweight='medium')
        ax1.set_ylabel('Effet sur le taux de mention\n(points de pourcentage)', fontweight='medium')
        ax1.set_title('A. Effet de la présence d\'au moins une bibliothèque', 
                    fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.4, linestyle='--')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.set_xlim(0.2, 5.5)

        # Annotations pour chaque point
        for i, (x, y) in enumerate(zip(rayons, coef_presence)):
            ax1.annotate(f'  {y:.3f}', xy=(x, y), xytext=(5, 0), 
                        textcoords='offset points', ha='left', va='center',
                        fontsize=9, color='#2c3e50')

        # Graphique 2 : Effet de densité (décroissant)
        ax2.errorbar(rayons, coef_densite, yerr=conf_densite,
                    fmt='s', color='#d62728', capsize=6, capthick=1.5,
                    linewidth=1.5, markersize=8, label='Estimation ponctuelle')

        # Courbe de tendance exponentielle pour illustrer la décroissance
        from scipy.optimize import curve_fit

        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c

        popt, _ = curve_fit(exp_decay, rayons, coef_densite, p0=[0.4, 1, 0])
        ax2.plot(rayons_smooth, exp_decay(rayons_smooth, *popt), '--', 
                color='#d62728', alpha=0.7, linewidth=1.5, label='Décroissance exponentielle')

        ax2.set_xlabel('Rayon (km)', fontweight='medium')
        ax2.set_ylabel('Effet par bibliothèque supplémentaire\n(points de pourcentage)', 
                    fontweight='medium')
        ax2.set_title('B. Effet du nombre de bibliothèques', 
                    fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.4, linestyle='--')
        ax2.legend(loc='best', framealpha=0.9)
        ax2.set_xlim(0.2, 5.5)

        # Annotations pour chaque point
        for i, (x, y) in enumerate(zip(rayons, coef_densite)):
            ax2.annotate(f'  {y:.3f}', xy=(x, y), xytext=(5, 0),
                        textcoords='offset points', ha='left', va='center',
                        fontsize=9, color='#2c3e50')

        # Ajout d'une note méthodologique en bas de la figure
        fig.text(0.02, 0.02, 
                'Note : Les barres d\'erreur représentent les intervalles de confiance à 95%.\n'
                'Les courbes en pointillés illustrent les tendances qualitatives.',
                fontsize=9, style='italic', alpha=0.7)

        # Ajustement de l'espacement
        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        plt.show()
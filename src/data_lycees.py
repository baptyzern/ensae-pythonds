import os
from src.download_file import download_file
import pandas as pd
import geopandas as gpd


def get_data_lycees():

    # Téléchargement des fichiers         ----------------------------------------------------------

    link_api_depp = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"

    if not os.path.exists("data/"):
        os.mkdir("data/")

    # Données sur les résultats         ------------------------------------------------------------

    url_lycees_resultats = link_api_depp + "fr-en-indicateurs-de-resultat-des-lycees-gt_v2/exports/parquet?lang=fr&timezone=Europe%2FBerlin"

    if not os.path.exists("data/lycees_resultats.parquet"):
        download_file(url_lycees_resultats, "data/lycees_resultats.parquet")

    lycees_resultats = pd.read_parquet(
        path="data/lycees_resultats.parquet",
        columns=[
            'uai', 'annee',
            # 'libelle_uai',
            # 'code_region', 'libelle_region',
            # 'libelle_academie',
            # 'code_departement', 'libelle_departement',
            # 'code_commune', 'libelle_commune',
            # 'secteur'

            # 'eff_2nde', 'eff_1ere', 'eff_term',
            # 'presents_total',
            # 'presents_l', 'presents_es', 'presents_s',
            'presents_gnle',
            # 'presents_sti2d', 'presents_std2a', 'presents_stmg', 'presents_stl'
            # 'presents_st2s', 'presents_s2tmd', 'presents_sthr',
            # 'nb_mentions_tb_avecf_g', 'nb_mentions_tb_sansf_g',
            # 'nb_mentions_b_g', 'nb_mentions_ab_g',
            # 'nb_mentions_tb_avecf_t', 'nb_mentions_tb_sansf_t',
            # 'nb_mentions_b_t', 'nb_mentions_ab_t',
            # 'taux_acces_2nde', 'va_acces_2nde',
            # 'taux_acces_1ere', 'va_acces_1ere',
            # 'taux_acces_term', 'va_acces_term'

            # 'taux_reu_total', 'va_reu_total',
            # 'taux_reu_l', 'taux_reu_es', 'taux_reu_s',
            'taux_reu_gnle',
            # 'taux_reu_sti2d', 'taux_reu_std2a', 'taux_reu_stmg', 'taux_reu_stl',
            # 'taux_reu_st2s', 'taux_reu_s2tmd', 'taux_reu_sthr',
            # 'va_reu_l', 'va_reu_es', 'va_reu_s',
            'va_reu_gnle',
            # 'va_reu_sti2d', 'va_reu_std2a', 'va_reu_stmg', 'va_reu_stl',
            # 'va_reu_st2s', 'va_reu_s2tmd', 'va_reu_sthr'
            # 'taux_men_l', 'taux_men_es', 'taux_men_s',
            'taux_men_gnle',
            # 'va_men_l', 'va_men_es', 'va_men_s',
            'va_men_gnle',
            # 'taux_men_sti2d', 'taux_men_std2a', 'taux_men_stmg', 'taux_men_stl',
            # 'taux_men_st2s', 'taux_men_s2tmd', 'taux_men_sthr',
            # 'va_men_sti2d', 'va_men_std2a', 'va_men_stmg', 'va_men_stl',
            # 'va_men_st2s', 'va_men_s2tmd', 'va_men_sthr'
        ]
        )

    # Changement des types des variables
    lycees_resultats['presents_gnle'] = pd.to_numeric(lycees_resultats['presents_gnle'], errors='coerce')
    lycees_resultats['taux_reu_gnle'] = pd.to_numeric(lycees_resultats['taux_reu_gnle'], errors='coerce')
    lycees_resultats['va_reu_gnle'] = pd.to_numeric(lycees_resultats['va_reu_gnle'], errors='coerce')
    lycees_resultats['taux_men_gnle'] = pd.to_numeric(lycees_resultats['taux_men_gnle'], errors='coerce')
    lycees_resultats['va_men_gnle'] = pd.to_numeric(lycees_resultats['va_men_gnle'], errors='coerce')

    lycees_resultats['annee'] = lycees_resultats['annee'].astype(str).str[0:4].astype(int)
    lycees_resultats = lycees_resultats.sort_values(["uai", "annee"])
    lycees_resultats = lycees_resultats.reset_index().drop(columns='index')

    # Données sur les IPS         ------------------------------------------------------------------

    url_lycees_ips = link_api_depp + "fr-en-ips-lycees-ap2023/exports/parquet?lang=fr&timezone=Europe%2FBerlin"

    if not os.path.exists("data/lycees_ips.parquet"):
        download_file(url_lycees_ips, "data/lycees_ips.parquet")

    lycees_ips = pd.read_parquet(
        path="data/lycees_ips.parquet",
        columns=[
            'uai', 'rentree_scolaire',
            # 'nom_de_l_etablissement',
            # 'secteur', 'type_de_lycee',
            # 'code_region', 'region_academique',
            # 'code_academie', 'academie',
            # 'code_du_departement', 'departement',
            # 'code_insee_de_la_commune', 'nom_de_la_commune',

            'ips_voie_gt',
            # 'ips_voie_pro', 'ips_post_bac', 'ips_etab',
            'ecart_type_voie_gt',
            # 'ecart_type_voie_pro', 'ecart_type_etablissement',
            # 'ips_national_legt', # 'ips_national_lpo', 'ips_national_lp',
            # 'ips_national_legt_prive', 'ips_national_legt_public',
            # 'ips_national_lpo_prive', 'ips_national_lpo_public',
            # 'ips_national_lp_prive', 'ips_national_lp_public',
            # 'ips_academique_legt', 'ips_academique_lpo', 'ips_academique_lp',
            # 'ips_academique_legt_prive', 'ips_academique_legt_public',
            # 'ips_academique_lpo_prive', 'ips_academique_lpo_public',
            # 'ips_academique_lp_prive', 'ips_academique_lp_public',
            # 'ips_departemental_legt', 'ips_departemental_lpo', 'ips_departemental_lp',
            # 'ips_departemental_legt_prive', 'ips_departemental_legt_public',
            # 'ips_departemental_lpo_prive', 'ips_departemental_lpo_public',
            # 'ips_departemental_lp_prive', 'ips_departemental_lp_public'
        ]
        )

    lycees_ips['ips_voie_gt'] = pd.to_numeric(lycees_ips['ips_voie_gt'], errors='coerce')
    lycees_ips['ecart_type_voie_gt'] = pd.to_numeric(lycees_ips['ecart_type_voie_gt'], errors='coerce')

    lycees_ips['annee'] = lycees_ips['rentree_scolaire'].str[5:9].astype(int)
    lycees_ips = lycees_ips.drop(columns="rentree_scolaire")
    lycees_ips = lycees_ips.sort_values(["uai", "annee"])
    lycees_ips = lycees_ips.reset_index().drop(columns='index')

    # Données de l'annuaire de l'éducation         -------------------------------------------------

    url_annuaire_education = link_api_depp + "fr-en-annuaire-education/exports/parquet?lang=fr&timezone=Europe%2FBerlin"

    if not os.path.exists("data/annuaire_education.parquet"):
        download_file(url_annuaire_education, "data/annuaire_education.parquet")

    annuaire_education = gpd.read_parquet(
        path="data/annuaire_education.parquet",
        columns=[
            'identifiant_de_l_etablissement',
            'nom_etablissement',
            'position', 'latitude', 'longitude',
            # 'telephone', 'fax', 'web', 'mail',
            # 'adresse_1', 'adresse_2', 'adresse_3', 'code_postal',
            'code_commune', 'code_departement', 'code_academie', 'code_region',
            'nom_commune', 'libelle_departement', 'libelle_academie', 'libelle_region',
            # 'code_circonscription', 'nom_circonscription',
            # 'siren_siret', 'nombre_d_eleves', 'fiche_onisep',
            # 'coordx_origine', 'coordy_origine', 'epsg_origine',
            # 'precision_localisation',

            # 'date_ouverture', 'date_maj_ligne', 'etat', 'ministere_tutelle',
            # 'multi_uai', 'code_type_contrat_prive',
            # 'code_nature', 'libelle_nature',
            # 'etablissement_mere', 'type_rattachement_etablissement_mere',
            # 'code_zone_animation_pedagogique', 'libelle_zone_animation_pedagogique',
            # 'code_bassin_formation', 'libelle_bassin_formation'
            'type_etablissement', 'statut_public_prive',
            # 'type_contrat_prive',
            # 'ecole_maternelle', 'ecole_elementaire',
            # 'rpi_concentre', 'rpi_disperse',

            'restauration', 'hebergement',
            # 'ulis', 'apprentissage', 'segpa', 'appartenance_education_prioritaire',
            # 'greta', 'pial',

            'voie_generale', 'voie_technologique', 'voie_professionnelle',
            'section_arts', 'section_cinema', 'section_theatre', 'section_sport',
            'section_internationale', 'section_europeenne',
            'lycee_agricole', 'lycee_militaire', 'lycee_des_metiers', 'post_bac',
            ]
        )
    # Sauvegarde du CRS dans une variable à côté pour plus tard
    annuaire_crs = annuaire_education.crs

    # Avoir des noms de variables cohérents entre bases
    annuaire_education = annuaire_education.rename(columns={
        'identifiant_de_l_etablissement': 'uai',
        'nom_etablissement': 'libelle_etablissement',
        })

    # Filtre pour n'avoir qu'une ligne par lycée
    annuaire_education['is_annexe'] = annuaire_education['libelle_etablissement'].str.contains('annexe')
    annuaire_education = annuaire_education[~annuaire_education['is_annexe']]
    annuaire_education = annuaire_education.drop(columns='is_annexe')

    # Certains établissements sont en double (car rattaché à plusieurs communes qui ont fusionné)
    annuaire_education = annuaire_education.groupby('uai').first().reset_index()

    # Rajout de l'information sur la densité de la commune
    commune_url = "https://object.files.data.gouv.fr/hydra-parquet/hydra-parquet/1f4841ac6cc0313803cabfa2c7ca4d37.parquet"

    if not os.path.exists("data/annuaire_communes.parquet"):
        download_file(commune_url, "data/annuaire_communes.parquet")
    annuaire_communes = pd.read_parquet(
        'data/annuaire_communes.parquet',
        columns=[
            # 'Unnamed: 0',
            'code_insee',
            # 'nom_standard', 'nom_sans_pronom', 'nom_a', 'nom_de',
            # 'nom_sans_accent', 'nom_standard_majuscule', 'typecom', 'typecom_texte', 'reg_code',
            # 'reg_nom', 'dep_code', 'dep_nom', 'canton_code', 'canton_nom', 'epci_code', 'epci_nom',
            # 'academie_code', 'academie_nom', 'code_postal', 'codes_postaux', 'zone_emploi',
            # 'code_insee_centre_zone_emploi', 'code_unite_urbaine', 'nom_unite_urbaine',
            # 'taille_unite_urbaine', 'type_commune_unite_urbaine', 'statut_commune_unite_urbaine',
            # 'population', 'superficie_hectare', 'superficie_km2', 'densite', 'altitude_moyenne',
            # 'altitude_minimale', 'altitude_maximale', 'latitude_mairie', 'longitude_mairie',
            # 'latitude_centre', 'longitude_centre',
            # 'grille_densite',
            'grille_densite_texte',
            # 'niveau_equipements_services', 'niveau_equipements_services_texte', 'gentile',
            # 'url_wikipedia'
            ]
        )

    annuaire_communes = annuaire_communes.rename(columns={
        'code_insee': 'code_commune',
        })

    annuaire_education = annuaire_education.merge(
        annuaire_communes,
        on='code_commune',
        how='left'
    )

    # Le code commune utilisé dans l'annuaire des communes ne prend pas
    # en compte l'arrondissement (contrairement au code commune dans
    # l'annuaire de l'éducation) -> correction pour Paris, Lyon et Marseille
    etab_ville_arr = annuaire_education['nom_commune'].isin(['Marseille', 'Paris', 'Lyon'])
    annuaire_education.loc[etab_ville_arr, 'grille_densite_texte'] = 'Grands centres urbains'

    # Transformations en variables catégorielles
    annuaire_education['grille_densite_texte'] = pd.Categorical(
        annuaire_education['grille_densite_texte'],
        ordered=True,
        categories=[
            'Grands centres urbains',
            'Centres urbains intermédiaires',
            'Petites villes',
            'Ceintures urbaines',
            'Bourgs ruraux',
            'Rural à habitat dispersé',
            'Rural à habitat très dispersé'  # AUCUN LYCEE CONCERNE
        ]
    )
    # En raison des faibles effectifs, on agrège certaines catégories 
    annuaire_education['grille_densite_4'] = (
        annuaire_education['grille_densite_texte']
        .replace({'Rural à habitat dispersé': 'Bourgs ruraux'})
        .replace({'Ceintures urbaines': 'Petites villes'})
        .cat.remove_unused_categories()
    )
    annuaire_education['statut_public_prive'] = pd.Categorical(
        annuaire_education['statut_public_prive'],
        ordered=True,
        categories=[
            'Public',
            'Privé'
        ]
    )

    # Perte du CRS avec le groupby.first (bug ?)
    annuaire_education.crs = annuaire_crs

    return lycees_resultats, lycees_ips, annuaire_education


def merge_data_lycees(lycees_resultats, lycees_ips, annuaire_education):
    # Restriction aux résultats de 2024 (année scolaire 2023-2024)

    lycees_resultats = lycees_resultats[lycees_resultats['annee'] == 2024]
    lycees_ips = lycees_ips[lycees_ips['annee'] == 2024]

    # Merge des sources de données Résultats et IPS
    # La "bonne source" de données est celle des résultats
    # Elle concerne les lycées généraux (et technologiques), alors que
    # Dans la base IPS, il y a aussi les lycées uniquement professionnels
    lycees_data_0 = pd.merge(
        lycees_resultats,
        lycees_ips,
        on=['uai', 'annee'],
        how='inner'
    )
    # On perd 9 lycées (?)

    # Ajout de la localisation
    lycees_data = pd.merge(
        lycees_data_0,
        annuaire_education,
        on=['uai'],
        how='inner'
    )
    # On perd 3 lycées (?)

    # Re-conversion en gdf, et changement de CRS
    lycees_data = gpd.GeoDataFrame(lycees_data, geometry='position', crs=annuaire_education.crs)
    lycees_data = lycees_data.to_crs(epsg=2154)

    return lycees_data


def filter_data_lycees(lycees_data):
    # Filtre pour sélectionner les lycées ayant suffisamment de candidats
    seuil_effectifs = 50
    lycees_data = lycees_data[lycees_data['presents_gnle'] >= seuil_effectifs]

    # Filtre pour sélectionner les lycées situés dans l'Hexagone
    regions_hors_hexagone = [
        '00', '01', '02', '03', '04', '06',
        '94'  # Corse
        ]
    lycees_data = lycees_data[~lycees_data['code_region'].isin(regions_hors_hexagone)]
    lycees_data = lycees_data.reset_index()

    return lycees_data

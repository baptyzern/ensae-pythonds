import os
from src.download_parquet import download_parquet
import pandas as pd
import geopandas as gpd


def get_data_biblio():

    # Téléchargement des fichiers               ----

    url_equipements = "https://www.insee.fr/fr/statistiques/fichier/8217525/BPE24.parquet"
    if not os.path.exists("data/"):
        os.mkdir("data/")
    if not os.path.exists("data/bpe_equipements.parquet"):
        download_parquet(url_equipements, "data/bpe_equipements.parquet")
    bpe_equipements = pd.read_parquet(
        path="data/bpe_equipements.parquet",
        columns=[
            "AN", "NOMRS",
            # "CNOMRS", "NUMVOIE", "INDREP", "TYPVOIE", "LIBVOIE", "CADR",
            "CODPOS", "DEPCOM", "DEP", "REG", "LIBCOM",
            # "DOM", "SDOM",
            "TYPEQU",
            # "SIRET", "STATUT_DIFFUSION", "CANTINE", "INTERNAT", "RPI", "EP", "CL_PGE", "SECT",
            # "ACCES_AIRE_PRATIQUE", "ACCES_LIBRE", "ACCES_SANITAIRE", "ACCES_VESTIAIRE",
            # "CAPACITE_D_ACCUEIL", "PRES_DOUCHE", "PRES_SANITAIRE", "SAISONNIER",
            # "COUVERT", "ECLAIRE",
            "CATEGORIE",
            # "MULTIPLEXE", "STRUCTURE_EXERCICE", "SPECIALITE", "ACCUEIL", "ITINERANCE",
            # "MODE_GESTION", "SSTYPHEB", "TYPE", "IMPLANTATION_STATION", "CAPACITE", "INDIC_CAPA",
            # "NBEQUIDENT", "INDIC_NBEQUIDENT", "NBSALLES", "INDIC_NBSALLES", "NBLIEUX",
            # "INDIC_NBLIEUX", "NB_PDC", "INDIC_NB_PDC", "NB_PDC_PA", "INDIC_NB_PDC_PA",
            # "NB_PDC_ACCELEREE", "INDIC_NB_PDC_ACCELEREE", "NB_PDC_LENTE", "INDIC_NB_PDC_LENTE",
            # "NB_PDC_RAPIDE", "INDIC_NB_PDC_RAPIDE", "NB_PDC_ULTRARAPIDE",
            # "INDIC_NB_PDC_ULTRARAPIDE", "NB_JOURS_OUVERT", "INDIC_NB_JOURS_OUVERT",
            "LAMBERT_X", "LAMBERT_Y", "LONGITUDE", "LATITUDE", "QUALITE_XY", "EPSG",
            # "QUALITE_GEOLOC", "TR_DIST_PRECISION", "DCIRIS", "QUALI_IRIS", "IRISEE", "QP2024",
            # "QUALI_QP2024", "QVA", "QUALI_QVA", "ZUS", "QUALI_ZUS", "EPCI",
            # "UU2020", "BV2022", "AAV2020", "DENS3", "DENS7"
        ]
        )
    bpe_bibliotheques = bpe_equipements[bpe_equipements['TYPEQU'] == "F307"]

    biblio_data = gpd.GeoDataFrame(
        data=bpe_bibliotheques,
        geometry=gpd.points_from_xy(
            x=bpe_bibliotheques['LAMBERT_X'],
            y=bpe_bibliotheques['LAMBERT_Y'],
            crs="EPSG:2154"
        )
    )

    return biblio_data

import os
from src.download_parquet import download_parquet
import pandas as pd
import geopandas as gpd


def get_data_lycees():

    # Téléchargement des fichiers               ----

    link_api_depp = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets"
    url_lycees_resultats = link_api_depp + "/fr-en-indicateurs-de-resultat-des-lycees-gt_v2/exports/parquet?lang=fr&timezone=Europe%2FBerlin"
    url_lycees_ips = link_api_depp + "/fr-en-ips-lycees-ap2023/exports/parquet?lang=fr&timezone=Europe%2FBerlin"
    url_lycees_geoloc = link_api_depp + "/fr-en-adresse-et-geolocalisation-etablissements-premier-et-second-degre/exports/parquet?lang=fr&timezone=Europe%2FBerlin"

    if not os.path.exists("data/"):
        os.mkdir("data/")

    if not os.path.exists("data/lycees_resultats.parquet"):
        download_parquet(url_lycees_resultats, "data/lycees_resultats.parquet")
    if not os.path.exists("data/lycees_ips.parquet"):
        download_parquet(url_lycees_ips, "data/lycees_ips.parquet")
    if not os.path.exists("data/lycees_geoloc.parquet"):
        download_parquet(url_lycees_geoloc, "data/lycees_geoloc.parquet")

    lycees_resultats = pd.read_parquet("data/lycees_ips.parquet")
    lycees_ips = pd.read_parquet("data/lycees_ips.parquet")
    lycees_geoloc = gpd.read_parquet("data/lycees_geoloc.parquet")

    return lycees_resultats, lycees_ips, lycees_geoloc

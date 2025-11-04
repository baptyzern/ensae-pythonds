import requests
import os

if not os.path.exists("data/"):
    os.mkdir("data/")

# URL des fichiers Parquet
va_url = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-indicateurs-de-resultat-des-lycees-gt_v2/exports/parquet?lang=fr&timezone=Europe%2FBerlin"
ips_url = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-ips_lycees/exports/parquet?lang=fr&timezone=Europe%2FBerlin"
bpe_url = "https://www.insee.fr/fr/statistiques/fichier/8217525/BPE24.parquet"


# Fonction pour télécharger un fichier Parquet
def download_parquet(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Échec du téléchargement du fichier `{filename}`")

# Téléchargement des fichiers
download_parquet(va_url, "data/lycees_resultats.parquet")
download_parquet(ips_url, "data/lycees_ips.parquet")
download_parquet(bpe_url, "data/bpe_equipements.parquet")

import os
from src.download_file import download_file
import pandas as pd
import geopandas as gpd
import xml.etree.ElementTree as ET


def get_xml_value(element, path, attribute=None):
    current = element
    for part in path.split('/'):
        if current is None:
            return None
        current = current.find(part)
    if current is None:
        return None
    if attribute:
        return current.get(attribute)
    return current.text


def extraire_xml_biblio(file):
    # Ouvrir le fichier XML
    tree = ET.parse(file)
    root = tree.getroot()

    # Initialiser une liste pour stocker les données extraites
    bibliotheques = []

    # Boucle for pour extraire sur chaque institution
    for institution in root.findall(".//INSTITUTION"):
        fields = institution.find('XML_FIELD')

        temp = {}

        temp["NOMETABLISSEMENT"] = get_xml_value(fields, "DID_SEL/NOMETABLISSEMENT")
        temp["ADRES_PAYS"] = get_xml_value(fields, "ADRESSES_SEL/ADRESSE_SET/ADRES_PAYS")
        temp["ADRES_CODEPOSTAL"] = get_xml_value(fields, "ADRESSES_SEL/ADRESSE_SET/ADRES_CODEPOSTAL")
        temp["ADRES_LATITUDE"] = get_xml_value(fields, "ADRESSES_SEL/ADRESSE_SET/ADRES_LATITUDE")
        temp["ADRES_LONGITUDE"] = get_xml_value(fields, "ADRESSES_SEL/ADRESSE_SET/ADRES_LONGITUDE")
        # temp["HEURESOUVERTURE"] = get_xml_value(fields, "ACCES_SEL/HEURESOUVERTURE")

        # temp["CONDITIONACCES"] = get_xml_value(fields, "ACCES_SEL/CONDITIONACCES")
        temp["CONDITIONACCES_d"] = get_xml_value(fields, "ACCES_SEL/CONDITIONACCES", "display")

        # temp["TYPEETABABES"] = get_xml_value(fields, "DID_SEL/TYPEETABABES")
        temp["TYPEETABABES_d"] = get_xml_value(fields, "DID_SEL/TYPEETABABES", "display")

        # temp["TYPEFAMABES"] = get_xml_value(fields, "DID_SEL/TYPEFAMABES")
        temp["TYPEFAMABES_d"] = get_xml_value(fields, "DID_SEL/TYPEFAMABES", "display")

        # temp["TYPEINST"] = get_xml_value(fields, "DID_SEL/TYPEINST")
        temp["TYPEINST_d"] = get_xml_value(fields, "DID_SEL/TYPEINST", "display")

        bibliotheques.append(temp)

    # Transformation en DataFrame
    df = pd.DataFrame(bibliotheques)

    return df


def get_data_biblio():

    # Téléchargement des fichiers               ----

    url_ccfr_biblio = "https://transfert.bnf.fr/link/02d946f4-138c-4f86-a075-1d9fde89169f"
    if not os.path.exists("data/"):
        os.mkdir("data/")
    if not os.path.exists("data/ccfr_biblio.xml"):
        download_file(url_ccfr_biblio, "data/ccfr_biblio.xml")

    df = extraire_xml_biblio("data/ccfr_biblio.xml")

    # Mettre en forme les latitudes et longitudes
    df['ADRES_LATITUDE'] = df['ADRES_LATITUDE'].str.replace(r'[^\d.]', '', regex=True)
    df['ADRES_LATITUDE'] = pd.to_numeric(df['ADRES_LATITUDE'])
    df['ADRES_LONGITUDE'] = df['ADRES_LONGITUDE'].str.replace(r'[^\d.]', '', regex=True)
    df['ADRES_LONGITUDE'] = pd.to_numeric(df['ADRES_LONGITUDE'])

    # Convertir en GeoDataFrame
    biblio_data = gpd.GeoDataFrame(
        data=df,
        geometry=gpd.points_from_xy(
            x=df['ADRES_LONGITUDE'],
            y=df['ADRES_LATITUDE'],
            crs="WGS 84"
        )
    )
    biblio_data = biblio_data.to_crs(epsg=2154)

    return biblio_data

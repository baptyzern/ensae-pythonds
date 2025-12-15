import geopandas as gpd


def calcul_distance(lycees, biblio):
    # Projection en Lambert 93
    lyc = lycees.to_crs("EPSG:2154")
    bib = biblio.to_crs("EPSG:2154")

    # Jointure spatiale nearest
    lyc_proches = gpd.sjoin_nearest(
        lyc,
        bib,
        how="left",
        distance_col="dist_proche_biblio_m"
    )

    lyc_proches['dist_proche_biblio_m'] = lyc_proches['dist_proche_biblio_m'].round()

    return lyc_proches

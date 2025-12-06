import geopandas as gpd

def calcul_biblio_rayons(lycees, biblio, rayons=(500, 1000, 2000, 5000)):
    # Projection
    lyc = lycees.to_crs("EPSG:2154")
    bib = biblio.to_crs("EPSG:2154")

    # DataFrame final
    result = lyc.copy()

    # Comptage des bibliothèques dans les rayons
    for r in rayons:

        # Création des buffers
        buffer_gdf = lyc[["uai", "position"]].copy()
        buffer_gdf["position"] = buffer_gdf.buffer(r)

        # Jointure spatiale : bibliothèques dans le rayon
        join = gpd.sjoin(
            bib,
            buffer_gdf,
            how="inner",
            predicate="within"
        )

        # Comptage par UAI
        counts = join.groupby("uai").size().rename(f"nb_biblio_{r}")

        # Fusion dans le dataframe
        result = result.merge(
            counts,
            left_on="uai",
            right_index=True,
            how="left"
        )

        result[f"nb_biblio_{r}"] = result[f"nb_biblio_{r}"].fillna(0)

    return result

import geopandas as gpd

def compute_biblio_distances(lycees, biblio,
                             rayons=(500, 1000, 2000, 5000)):

    lyc = lycees.to_crs(2154)

    bib = biblio.to_crs(2154)

    # Distance à la bibliothèque la plus proche
    lyc_proches = gpd.sjoin_nearest(
        lyc,
        bib,
        how="left",
        distance_col="dist_proche_biblio_m"
    )

    # Comptage des bibliothèques dans les rayons
    for r in rayons:
        
    # Créer un buffer autour de chaque lycée
      buffer_gdf = lyc[["uai", "position"]].copy()
      buffer_gdf["geometry"] = buffer_gdf.buffer(r)
    
    # Spatial join : bibliothèques dans le buffer
    join = gpd.sjoin(bib, buffer_gdf, how="inner", predicate="within")
    
    # Compter le nombre de bibliothèques par lycée
    counts = join.groupby("uai").size().rename(f"nb_biblio_{r}")
    
    # Fusion avec le GeoDataFrame lycées
    lyc = lyc.merge(counts, left_on="uai", right_index=True, how="left")
    
    lyc[f"nb_biblio_{r}"] = lyc[f"nb_biblio_{r}"].fillna(0)

    return lyc

import folium
import folium.plugins


def map_croisee(biblio_dep, lycees_dep, zoom_start=12):

    center = [lycees_dep.latitude.mean(), lycees_dep.longitude.mean()]

    m = folium.Map(
        location=center,
        zoom_start=zoom_start
        )

    tooltip = folium.GeoJsonTooltip(
        fields=['NOMETAB', 'TYPEETABABES_d', 'CONDITIONACCES_d'],
        aliases=['Nom :', 'TYPEETABABES : ', 'CONDITIONACCES_d :'],
        localize=True
    )
    folium.GeoJson(
        biblio_dep,
        tooltip=tooltip,
        marker=folium.Marker(icon=folium.Icon(color="green")),
    ).add_to(m)

    tooltip = folium.GeoJsonTooltip(
        fields=[
            'libelle_etablissement', 'statut_public_prive', 'presents_gnle',
            'taux_reu_gnle', 'taux_men_gnle', 'ips_voie_gt'
            ],
        aliases=[
            'Nom :', 'Secteur :', 'Nombre de candidats présents :',
            'Taux de réussite :', 'Taux de mention', 'Indice de position sociale :'
        ],
        localize=True
    )
    folium.GeoJson(
        lycees_dep,
        tooltip=tooltip,
        marker=folium.Marker(icon=folium.Icon(color="blue")),
    ).add_to(m)

    folium.plugins.ScrollZoomToggler().add_to(m)

    return m

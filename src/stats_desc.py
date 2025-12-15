import folium
import folium.plugins
import pandas as pd


def map_croisee(biblio_dep, lycees_dep, zoom_start=12):

    center = [lycees_dep.latitude.mean(), lycees_dep.longitude.mean()]

    m = folium.Map(
        location=center,
        zoom_start=zoom_start
        )

    tooltip = folium.GeoJsonTooltip(
        fields=['NOMETAB', 'TYPEFAMABES_d'],
        aliases=['Nom :', "Type de bibliothèque : "],
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
            'taux_reu_gnle', 'taux_men_gnle', 'ips_voie_gt', 'dist_proche_biblio_m'
            ],
        aliases=[
            'Nom :', 'Secteur :', 'Nombre de candidats présents :',
            'Taux de réussite :', 'Taux de mention', 'Indice de position sociale :',
            'Distance à la bibliothèque la plus proche (m) : '
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


def lycees_stat_desc_num(lycees_data):
    lycees_stat_desc_num = lycees_data.melt(
        id_vars=['uai'],
        value_vars=[
            'taux_reu_gnle', 'va_reu_gnle',
            'taux_men_gnle', 'va_men_gnle',
            'ips_voie_gt', 'ecart_type_voie_gt'
        ]
    )

    lycees_stat_desc_num['variable'] = pd.Categorical(
        lycees_stat_desc_num['variable'],
        ordered=True,
        categories=[
            'taux_reu_gnle', 'taux_men_gnle', 'ips_voie_gt',
            'va_reu_gnle', 'va_men_gnle', 'ecart_type_voie_gt'
        ]
    ).rename_categories([
        'Taux de réussite (%)', 'Taux de mention (%)',
        'Indice de position sociale', 'Valeur ajoutée du\ntaux de réussite',
        'Valeur ajoutée du\ntaux de mention', 'Ecart-type de l\'IPS'
    ])

    return lycees_stat_desc_num

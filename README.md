# Quel est l'impact des bibliothèques à proximité des lycées sur la réussite des élèves ?

Ce projet a été réalisé dans le cadre du cours *Python pour la data science*
donné en deuxième année de l'ENSAE.

*Décembre 2025*\
*Auteurs : Yacouba Biyen, Nada Maslouhi, Baptiste Yzern*

## Abstract

Les performances des lycées sont particulièrement scrutés à partir des résultats à l'examen
du baccalauréat. Le taux de mention discrimine particulièrement les bons lycées des mauvais.
Ces performances sont le plus souvent expliquées par le profil social des élèves accueillis
dans l'établissement. Est-il cependant possible de trouver d'autres facteurs de réussite des
élèves ? \
Notre travail propose ainsi d'étudier l'impact des bibliothèques sur le taux de mention au 
baccalauréat général des lycées ayant envoyé au moins 50 candidats. Le champ est restreint
à l'année 2024 et aux lycées situés dans l'Hexagone (Corse exclue).\
Pour étudier ce lien, nous mobilisons principalement les données publiques de la Depp (le 
service statistique ministériel de l'Education nationale). Nous les enrichissons à partir
d'une part de données sur la densité des communes (données disponibles sur data.gouv.fr), et 
d'autre part à partir du Catalogue collectif de France piloté par la Bibliothèque nationale
de France. Ce dernier recense l'essentiel des bibliothèques de France de manière géolocalisée.

Plus de détails sont donnés dans le notebook qui sert de rapport final
([lien vers le notebook](https://github.com/baptyzern/ensae-pythonds/blob/main/final_notebook.ipynb)).

### Sources de données mobilisées

-   *Indicateurs de résultats des lycées* 
([lien direct vers la source](https://data.education.gouv.fr/explore/dataset/fr-en-indicateurs-de-resultat-des-lycees-gt_v2/information/))
-   *Indices de position sociale des lycées*, sur data.education.gouv.fr
([lien direct vers la source](https://data.education.gouv.fr/explore/dataset/fr-en-ips-lycees-ap2023/information/))
-   *Annuaire de l'éducation* 
([lien direct vers la source](https://data.education.gouv.fr/explore/dataset/fr-en-annuaire-education/information))
-   *Communes et villes de France* 
([lien direct vers la source](https://www.data.gouv.fr/datasets/communes-et-villes-de-france-en-csv-excel-json-parquet-et-feather/informations))
-   *Fiches descriptives de bibliothèques issues du répertoire du CCFr* 
([lien direct vers la source](https://api.bnf.fr/index.php/fr/CCFr/Repertoire_Bibliotheques))

## Considérations techniques

Le projet est entièrement reproductible à partir du notebook principal 
([lien vers le notebook](https://github.com/baptyzern/ensae-pythonds/blob/main/final_notebook.ipynb))

Les données sont téléchargées automatiquement à partir d'URL fixes.\
Le téléchargement se fait le plus souvent en moins d'une minute.

La liste des dépendances peut être installée depuis le terminal à l'aide de la commande suivante (attention à se placer dans le dossier du projet avant de l'exécuter) :

```
pip install -r requirements.txt
```

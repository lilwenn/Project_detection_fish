## Suivi et détection de poissons dans des vidéos

Analyse de comportement des poissons
Ce script Python utilise la détection d'objets et le suivi pour analyser le comportement des poissons dans des vidéos.
Il offre plusieurs fonctionnalités telles que la détection des positions des poissons, l'affichage des centres des groupes de poissons, la génération de heatmap,
le suivi des poissons, la détection du poisson le plus éloigné du groupe, le calcul de la vitesse de chaque poisson, l'identification des proches voisins,
et la segmentation en clusters avec les méthodes DBSCAN et KMEANS.

## Installation

1. Installez Ultralytics YOLO en utilisant la commande suivante : pip install ultralytics
2. Assurez-vous d'avoir les dépendances suivantes installées : OpenCV, pandas, Ultralytics, numpy, matplotlib, sklearn, etc...

## Utilisation

Installez les dépendances en utilisant la commande suivante : pip install -r requirements.txt
Chemin des fichiers d'entrée : Assurez-vous d'avoir les fichiers vidéo à analyser et spécifiez les chemins appropriés dans le script.

Exécution du script : Exécutez le script en utilisant la commande suivante : python Projet_M1_fiche.py

Choix des fonctionnalités : Suivez les instructions affichées pour choisir la fonctionnalité souhaitée dans le menu avec les numéros

Fonctionnalités disponibles :
- Entraînement du modèle : Permet d'entraîner un modèle pour la détection des poissons.
- Détection des positions des poissons sur une vidéo : Utilise le modèle YOLO pour détecter les poissons sur une vidéo.
- Affichage centre poisson : Affiche les positions des poissons sur la vidéo d'origine.
- Affichage des centres des groupes des poissons sur une vidéo : Identifie et affiche les centres des groupes de poissons sur la vidéo.
- Heatmap sur une vidéo : Génère une heatmap des mouvements des poissons sur la vidéo.
- Tracking sur une vidéo : Suit les poissons à travers la vidéo.
- Trouver le poisson le plus éloigné : Identifie le poisson le plus éloigné par rapport au groupe à chaque instant de la vidéo.
- Trouver la vitesse de chaque poisson : Estime la vitesse de déplacement de chaque poisson dans la vidéo.
- Proches voisins : Identifie les proches voisins de chaque poisson dans la vidéo.
- Affichage des proches voisins sur une vidéo : Visualise les proches voisins de chaque poisson sur la vidéo.
- Affichage des clusters avec méthode DBSCAN sur une vidéo : Segmente les poissons en groupes en utilisant l'algorithme DBSCAN et visualise les clusters sur la vidéo.
- Affichage des clusters avec méthode KMEANS sur une vidéo : Segmente les poissons en groupes en utilisant l'algorithme K-Means et visualise les clusters sur la vidéo.

## Contributions

Les contributions externes sont les bienvenues. Veuillez soumettre une demande de tirage pour proposer vos contributions.

## Auteurs

- LEA DENIEL https://github.com/leadnl
- LILWENN LE GAC https://github.com/lilwenn

## Ressources supplémentaires

- https://github.com/ultralytics/ultralytics/blob/main/README.md

## Contact

Pour toute question ou commentaire, veuillez contacter l.deniel@orange.fr et lilwenn.legac@gmail.com
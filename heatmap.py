# Impotation des bibliothèques

import cv2
from ultralytics.solutions import heatmap
import numpy as np


def generate_heatmap(model, video_path, output_image):
    """
    Génère une carte de chaleur à partir d'une vidéo d'entrée et enregistre l'image résultante.

    @input :
    - model : Le modèle de détection à utiliser.
    - video_path (str): Le chemin de la vidéo d'entrée.
    - output_image (str): Le chemin de l'image de sortie pour la carte de chaleur.

    @output :
    L'image de la carte de chaleur est sauvegardée.
    """

    # Lecture de la vidéo
    cap = cv2.VideoCapture(video_path)
    # Vérification si la vidéo est ouverte avec succès
    assert cap.isOpened(), "Erreur lors de la lecture du fichier vidéo"
    # Récupération de la largeur, hauteur et nombre d'images par seconde de la vidéo
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Initialisation de l'objet heatmap pour la génération de la carte de chaleur
    heatmap_obj = heatmap.Heatmap()
    heatmap_obj.set_args(colormap=cv2.COLORMAP_HOT,
                         imw=w,
                         imh=h,
                         view_img=True,
                         shape="circle")

    # Initialisation d'une variable pour stocker la somme normalisée des cartes de chaleur de toutes les frames
    heatmap_sum_normalized = np.zeros((h, w), dtype=np.float64)
    frame_count = 0

    # Boucle à travers les frames de la vidéo
    while cap.isOpened():
        # Lecture de la prochaine frame de la vidéo
        success, frame = cap.read()
        # Vérification si la lecture est réussie
        if not success:
            print("Le traitement de la vidéo a été terminé avec succès.")
            break
        # Suivi des objets dans la frame courante
        tracks = model.track(frame, persist=True, show=False)
        # Génération de la carte de chaleur pour la frame courante
        heatmap_frame = heatmap_obj.generate_heatmap(frame, tracks)
        # Normalisation de la carte de chaleur
        heatmap_frame_normalized = cv2.normalize(heatmap_frame, None, 0, 255, cv2.NORM_MINMAX)
        # Conversion de la carte de chaleur normalisée en niveau de gris
        heatmap_frame_gray = cv2.cvtColor(heatmap_frame_normalized, cv2.COLOR_RGB2GRAY)
        # Ajout de la carte de chaleur normalisée de la frame courante à la somme totale
        heatmap_sum_normalized += heatmap_frame_gray.astype(np.float64)
        frame_count += 1

    # Calcul de la carte de chaleur moyenne
    heatmap_avg_normalized = (heatmap_sum_normalized / frame_count).astype(np.uint8)

    # Enregistrement de la carte de chaleur moyenne en tant qu'image PNG
    cv2.imwrite(output_image, heatmap_avg_normalized)

    # Charger l'image en niveaux de gris
    gray_image = cv2.imread(output_image, cv2.IMREAD_GRAYSCALE)

    # Normaliser les valeurs de l'image en niveaux de gris entre 0 et 1
    normalized_image = gray_image / 255.0

    # Créer une colormap personnalisée allant du bleu au rouge
    custom_colormap = cv2.applyColorMap((normalized_image * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Afficher et enregistrer l'image
    cv2.imwrite(output_image, custom_colormap)
    cv2.destroyAllWindows()

    # Libération des ressources
    cap.release()

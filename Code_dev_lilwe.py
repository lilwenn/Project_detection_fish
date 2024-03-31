# Impotation des bibliothèques

from ultralytics import YOLO
from IPython.display import display, Image
import cv2
import pandas as pd
import json
from ultralytics.solutions import heatmap
import numpy as np
import matplotlib.pyplot as plt
import math
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import os
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


def model_training(current_dir):
    """
    Entraîne le modèle YOLO sur les données fournies.

    @input :
    - current_dir (str): Le chemin du répertoire contenant les données d'entraînement.
    """

    model_path = os.path.join(current_dir, 'yolov8m.pt')
    data_yaml_path = os.path.join(current_dir, 'data.yaml')
    trained_model_path = os.path.join(current_dir, 'best.pt')
    openvino_model_path = os.path.join(current_dir, 'best_openvino_model')

    # Entrainement du modèle
    model = YOLO(model_path)
    result = model.train(data=data_yaml_path, epochs=25, imgsz=640)

    # Charger la meilleure version du modèle
    model = YOLO(trained_model_path)

    # Exportation du modele
    model.export(format='openvino', export_path=openvino_model_path)


def video_detection(video_path, output_video_path, model):
    """
    Détecte et suit les poissons dans une vidéo à l'aide du modèle.

    @input :
    - video_path (str): Le chemin de la vidéo à traiter.
    - output_video_path (str): Le chemin de sortie de la vidéo traitée.
    - model (obj): Le modèle utilisé pour la détection des poissons.

    @output :
    la vidéo traitée est sauvegardée à output_video_path.

    """

    # Ouverture de la vidéo à traiter
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    out = None

    dico_position = {}  # dictionnaire avec les frames et les positions des poissons
    frame_index = 0

    # Boucle à travers les frames de la vidéo
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Faire le tracking
            results = model.track(frame, persist=True)

            dico_position[frame_index] = {}

            if results:
                for r in results:
                    if r is not None and r.boxes is not None:
                        i = 0
                        if r.boxes.id is not None:
                            for id in r.boxes.id:
                                dico_position[frame_index][id] = r.boxes.xyxy[i]
                                i += 1
                    else:
                        # Gérer le cas où aucune boîte n'a été détectée
                        pass

            frame_index += 1

            # Affichage de l'avancement de chaque frame dans la console
            print(f"Traitement de la frame {frame_index}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

            # Anotation des frames
            annotated_frame = results[0].plot()

            # Affichage du frame annoté
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Création d'un objet VideoWriter
            if out is None:
                out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                      (annotated_frame.shape[1], annotated_frame.shape[0]))

            # Enregistrement de la frame annotée dans la vidéo
            out.write(annotated_frame)

            # Arrêt si la touche q est pressée
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    if out is not None:
        out.release()

    return dico_position, fps


def fish_position_dico(dico_position):
    """
    Crée un dictionnaire des positions des poissons par frame.

    @input :
    - dico_position (dict): Un dictionnaire contenant les positions des poissons par frame.

    @output :
    - dico_position (dict): Le dictionnaire mis à jour des positions des poissons par frame.
    """

    fish_centers = {}
    for frame_num in dico_position.keys():
        fish_centers[frame_num] = {}
        for id in dico_position[frame_num].keys():
            fish_centers[frame_num][int(id.item())] = []
            positions = dico_position[frame_num][id]
            positions = positions.tolist()
            x1, y1, x2, y2 = positions[0], positions[1], positions[2], positions[3]
            center_x = float((x1 + x2) / 2)
            center_y = float((y1 + y2) / 2)
            fish_centers[frame_num][int(id.item())].append((center_x, center_y))
    return fish_centers


def draw_fish_centers(frame, centers):
    """
    Dessine les centres des poissons sur chaque frame d'une vidéo.

    @input :
    - frame (array): La frame de la vidéo.
    - centers (array): Les centres des poissons détectés.

    """

    for center in centers.values():
        center_x, center_y = center[0]  # Les coordonnées du centre sont dans la première (et unique) entrée de la liste
        center_x, center_y = int(center_x), int(center_y)
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)


def display_fish_centers(fish_centers, video_path, output_video_path, fps_output_file):
    """
    Affiche les centres des poissons sur chaque frame d'une vidéo.

    @input :
    - fish_centres (dict): Un dictionnaire des centres des poissons par frame.
    - video_path (str): Le chemin de la vidéo traitée.
    - output_video_path (str): Le chemin de sortie de la vidéo avec les centres des poissons.
    - fps_output_file (str): Le chemin de sortie du fichier vidéo avec les FPS.

    @output :
    la vidéo traitée est sauvegardée à output_video_path.
    """

    cap = cv2.VideoCapture(video_path)

    # Définir le codec vidéo et les fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    with open(fps_path, 'r') as f:
        fps = float(f.read())
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    frame_num = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Dessiner les centres des poissons sur la frame actuelle
            if str(frame_num) in fish_centers:
                draw_fish_centers(frame, fish_centers[str(frame_num)])

            # Écriture dans la vidéo de sortie
            out.write(frame)

            # Affichage de l'avancement de chaque frame dans la console
            print(f"Traitement de la frame {frame_num}/{total_frames}")

            # Affichage de la frame
            cv2.imshow('Video avec centres de poissons', frame)

            frame_num += 1

            # Arret si la touche q pressée
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # Release the video capture object and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Calcul du centre du groupe par frame
def calculate_group_center(fish_centers):
    """
    Calcule le centre du groupe de poissons pour chaque frame.

    @input :
    - fish_centers (dict): Un dictionnaire des centres des poissons par frame.

    @output :
    - group_center (tuple): Les coordonnées du centre du groupe de poissons.
    """

    group_centers = {}
    for frame_num, frame_data in fish_centers.items():
        if frame_data:
            x_sum = sum(center[0][0] for center in frame_data.values())
            y_sum = sum(center[0][1] for center in frame_data.values())
            num_centers = len(frame_data)
            group_center_x = x_sum / num_centers
            group_center_y = y_sum / num_centers
            group_centers[frame_num] = (group_center_x, group_center_y)
    return group_centers


def draw_group_center(frame, group_center):
    """
    Dessine le centre du groupe de poissons sur chaque frame d'une vidéo.

    @input :
    - frame (array): La frame de la vidéo.
    - group_center (tuple): Les coordonnées du centre du groupe de poissons.

    @output :
    - frame (array): La frame avec le centre du groupe de poissons dessiné.
    """

    if group_center is not None:
        center_x, center_y = group_center
        center_x, center_y = int(center_x), int(center_y)
        # Dessiner un carré bleu autour du centre du groupe
        cv2.rectangle(frame, (center_x - 10, center_y - 10), (center_x + 10, center_y + 10), (255, 0, 0), 2)


def calculate_distance_to_group(fish_centers, group_centers):
    """
    Calcule la distance entre chaque poisson et le centre du groupe de poissons pour chaque frame.

    @input :
    - fish_centers (dict): Un dictionnaire des centres des poissons par frame.
    - group_centers (list): Une liste des centres du groupe de poissons par frame.

    @output :
    - distances (list): Une liste des distances entre chaque poisson et le centre du groupe.
    """

    distances_per_frame = {}
    for frame_num, frame_data in fish_centers.items():
        group_center = group_centers[frame_num]
        distances_per_frame[frame_num] = {}
        for fish_id, fish_center in frame_data.items():
            # Les coordonnées du poisson sont dans fish_center[0]
            fish_x, fish_y = fish_center[0]
            # Calcul de la distance euclidienne entre le poisson et le centre du groupe
            distance = math.sqrt((fish_x - group_center[0]) ** 2 + (fish_y - group_center[1]) ** 2)
            distances_per_frame[frame_num][fish_id] = distance
    return distances_per_frame


def draw_lines_to_fish(frame, group_center, fish_centers):
    """
    Dessine des lignes reliant le centre du groupe à chaque poisson.

    @input :
    - frame (array): L'image de la frame vidéo.
    - group_center (tuple): Les coordonnées du centre du groupe.
    - fish_centers (dict): Un dictionnaire des centres de chaque poisson avec leurs identifiants.

    """

    if group_center is not None:
        for fish_id, fish_center in fish_centers.items():
            # Les coordonnées du poisson sont dans fish_center[0]
            fish_x, fish_y = fish_center[0]
            cv2.line(frame, (int(group_center[0]), int(group_center[1])), (int(fish_x), int(fish_y)), (255, 0, 0), 1)


def display_group_center(fish_centers, video_path, output_video_path):
    """
    Affiche le centre du groupe et sauvegarde la vidéo avec le centre du groupe

    @input :
    - fish_centers (dict): Un dictionnaire des centres de chaque poisson avec leurs identifiants.
    - video_path (str): Le chemin de la vidéo d'entrée.
    - output_video_path (str): Le chemin de la vidéo de sortie avec le centre du groupe


    @output :
    la vidéo traitée est sauvegardée à output_video_path.

    """
    group_centers = calculate_group_center(fish_centers)

    # Sauvegarder les centres de groupe dans un fichier JSON
    with open(group_centers_json_path, 'w') as f:
        json.dump(group_centers, f)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Lecture des FPS à partir du fichier
    with open(fps_path, 'r') as f:
        fps = float(f.read())
    out = cv2.VideoWriter(output_video_path, fourcc, fps,
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Nombre total de frames dans la vidéo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_num = 0
    while frame_num < total_frames:
        success, frame = cap.read()

        if success:
            # Dessiner le carré bleu sur le centre du groupe actuel
            if str(frame_num) in group_centers:
                draw_group_center(frame, group_centers[str(frame_num)])

            # Dessiner les traits vers chaque poisson
            if str(frame_num) in fish_centers and str(frame_num) in group_centers:
                draw_lines_to_fish(frame, group_centers[str(frame_num)], fish_centers[str(frame_num)])

            # Écriture dans la vidéo de sortie
            out.write(frame)

            # Affichage de l'avancement de chaque frame dans la console
            print(f"Traitement de la frame {frame_num}/{total_frames}")

            # Affichage de la frame
            cv2.imshow('Video avec carre bleu representant le centre du groupe', frame)

            frame_num += 1

            # Arrêt si la touche q est pressée
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # Libérer les ressources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


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

        # Affichage de l'avancement de chaque frame dans la console
        print(f"Traitement de la frame {frame_count}")

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


def calculate_distance(point1, point2):
    """
    Calcule la distance euclidienne entre deux points.

    @input :
    - point1 (tuple): Les coordonnées du premier point.
    - point2 (tuple): Les coordonnées du deuxième point.

    @output :
    - La distance entre les deux points.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Fonction pour trouver le poisson le plus éloigné du centre du groupe à chaque frame
def find_farthest_fish(fish_centers, group_centers):
    """
    Trouve le poisson le plus éloigné du centre du groupe à chaque frame.

    @input :
    - fish_centers (dict): Un dictionnaire des centres de chaque poisson avec leurs identifiants.
    - group_centers (dict): Un dictionnaire des centres du groupe pour chaque frame.

    @output :
    - farthest_fish_per_frame (dict): Un dictionnaire contenant l'identifiant du poisson le plus éloigné pour chaque frame.
    """

    farthest_fish_per_frame = {}
    for frame_num, frame_data in fish_centers.items():
        if str(frame_num) in group_centers:
            group_center = group_centers[frame_num]
            farthest_fish_index = None
            max_distance = -1
            for fish_id, fish_position in frame_data.items():
                distance = calculate_distance(fish_position[0], group_center)
                if distance > max_distance:
                    max_distance = distance
                    farthest_fish_index = fish_id
            farthest_fish_per_frame[frame_num] = farthest_fish_index
    return farthest_fish_per_frame


def count_farthest_fish(farthest_fish_per_frame, output_file):
    """
    Compte le nombre de fois que chaque poisson est le plus éloigné du centre du groupe.

    @input :
    - farthest_fish_per_frame (dict): Un dictionnaire contenant l'identifiant du poisson le plus éloigné pour chaque frame.
    - output_file (str): Chemin du fichier de sortie pour enregistrer les résultats.

    @output :
    - fish_counts (dict): Un dictionnaire contenant le nombre de fois que chaque poisson est le plus éloigné du centre du groupe.
    """

    fish_counts = {}
    for fish_id in farthest_fish_per_frame.values():
        fish_id = int(fish_id)  # Convertir la valeur en entier
        if fish_id in fish_counts:
            fish_counts[fish_id] += 1
        else:
            fish_counts[fish_id] = 1

    # Enregistrer les données dans le fichier de sortie au format JSON
    with open(output_file, 'w') as f:
        json.dump(fish_counts, f)

    return fish_counts


def draw_green_point(image, position):
    """
    Dessine un point vert sur une image à la position spécifiée.

    @input :
    - image (array): L'image sur laquelle dessiner le point.
    - position (tuple): Les coordonnées du point à dessiner.

    @output :
    Le point vert est dessiné sur l'image.
    """
    cv2.circle(image, (int(position[0]), int(position[1])), 5, (0, 255, 0), -1)

    # Écrire "isole" à côté du point vert
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner = (int(position[0]) + 10, int(position[1]) + 10)
    font_scale = 0.5
    font_color = (0, 255, 0)  # Couleur verte
    line_type = 2
    cv2.putText(image, 'isolation', bottom_left_corner, font, font_scale, font_color, line_type)


def draw_farthest_fish(fish_centers, farthest_fish_per_frame, video_path, output_video_path):
    """
    Dessine un point vert sur le poisson le plus éloigné du centre du groupe dans chaque frame de la vidéo.

    @input :
    - fish_centers (dict): Un dictionnaire des centres de chaque poisson avec leurs identifiants.
    - farthest_fish_per_frame (dict): Un dictionnaire indiquant pour chaque frame l'indice du poisson le plus éloigné.
    - video_path (str): Le chemin de la vidéo d'entrée.
    - output_video_path (str): Le chemin de la vidéo de sortie avec le poisson le plus éloigné.

    @output :
    La vidéo de sortie est sauvegardée avec le poisson le plus éloigné.
    """

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video_path, fourcc, fps,
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Nombre total de frames dans la vidéo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_num = 0
    while frame_num < total_frames:
        success, frame = cap.read()

        if success:
            # Dessiner un point vert au poisson le plus éloigné
            if str(frame_num) in farthest_fish_per_frame:
                indice_poisson_le_plus_eloigne = farthest_fish_per_frame[str(frame_num)]
                position_poisson_le_plus_eloigne = fish_centers[str(frame_num)][indice_poisson_le_plus_eloigne][0]
                draw_green_point(frame, position_poisson_le_plus_eloigne)

            # Afficher l'avancement de chaque frame dans la console
            print(f"Traitement de la frame {frame_num}/{total_frames}")

            # Écriture dans la vidéo de sortie
            out.write(frame)

            # Affichage de la frame
            cv2.imshow('Video avec carre bleu representant le centre du groupe', frame)
            frame_num += 1

            # Arrêt si la touche q est pressée
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # Libérer les ressources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def tracking(model, video_path, output_video):
    """
    Effectue le suivi des objets dans une vidéo à l'aide d'un modèle de détection spécifié.

    @input :
    - model : Le modèle de détection à utiliser.
    - video_path (str): Le chemin de la vidéo d'entrée.
    - output_video (str): Le chemin de la vidéo de sortie avec les possons.

    @output :
    La vidéo de sortie est sauvegardée avec les objets suivis.
    """
    track_history = defaultdict(lambda: [])
    model = YOLO("yolov8n-seg.pt")

    cap = cv2.VideoCapture(video_path)
    w, h = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))

    with open(fps_path, 'r') as f:
        fps = float(f.read())
    out = cv2.VideoWriter('instance-segmentation-object-tracking.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0

    while True:
        ret, im0 = cap.read()

        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        annotator = Annotator(im0, line_width=2)

        results = model.track(im0, persist=True)

        if results[0].boxes.id is not None and results[0].masks is not None:
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for mask, track_id in zip(masks, track_ids):
                annotator.seg_bbox(mask=mask,
                                   mask_color=colors(track_id, True),
                                   track_label=str(track_id))

        out.write(im0)
        cv2.imshow("instance-segmentation-object-tracking", im0)

        # Affichage de l'avancement de chaque frame dans la console
        print(f"Traitement de la frame {frame_num}/{total_frames}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1

    out.release()
    cap.release()
    cv2.destroyAllWindows()


def calculate_fish_speeds(fish_centers, fps):
    """
    Calcule les vitesses des poissons à partir de leurs centres dans chaque frame.

    @input :
    - fish_centers (dict): Un dictionnaire des centres de chaque poisson avec leurs identifiants.
    - fps (float): Le nombre d'images par seconde de la vidéo.

    @output :
    - fish_speeds (dict): Un dictionnaire contenant les vitesses des poissons pour chaque frame.
    """
    fish_speeds = {}
    prev_frame_centers = None

    for frame_index, frame_centers in sorted(fish_centers.items(), key=lambda x: int(x[0])):
        if prev_frame_centers is not None and len(frame_centers) == len(prev_frame_centers):
            speeds = []
            for fish_id, center in frame_centers.items():
                prev_center = prev_frame_centers.get(fish_id)
                if prev_center is not None:
                    distance = math.sqrt(
                        (center[0][0] - prev_center[0][0]) ** 2 + (center[0][1] - prev_center[0][1]) ** 2)
                    speed = distance * fps
                    speeds.append(speed)
            fish_speeds[frame_index] = speeds

        prev_frame_centers = frame_centers

    return fish_speeds


def plot_fish_speeds(fish_speeds, filename):
    """
    Trace et enregistre les courbes de vitesse des poissons.

    @input :
    - fish_speeds (dict): Un dictionnaire contenant les vitesses des poissons pour chaque frame.
    - filename (str): Le nom du fichier pour sauvegarder le graphique.

    @output :
    Le graphique est sauvegardé dans un fichier.
    """
    if not fish_speeds:
        print("Aucune donnée de vitesse de poisson disponible.")
        return

    num_fish = len(next(iter(fish_speeds.values())))
    if num_fish == 0:
        print("Aucun poisson trouvé dans les données de vitesse.")
        return

    fig, axes = plt.subplots(num_fish, 1, figsize=(10, 5 * num_fish))

    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]  # Convertir en liste si un seul sous-axe est créé

    for fish_index, ax in enumerate(axes):
        speeds = [speeds[fish_index] if fish_index < len(speeds) else 0 for speeds in fish_speeds.values()]
        ax.plot(list(fish_speeds.keys()), speeds, label=f'Poisson {fish_index + 1}')
        ax.set_xlabel('Trame')
        ax.set_ylabel('Vitesse (pixels par seconde)')
        ax.set_title(f'Courbe de vitesse du Poisson {fish_index + 1}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(filename)


def create_dict_neighbors(fish_centers, output):
    """
    Crée un dictionnaire des voisins de chaque poisson à chaque frame.

    @input :
    - fish_centers (dict): Un dictionnaire des centres de chaque poisson avec leurs identifiants.
    - output (str): Le chemin de sortie pour sauvegarder le dictionnaire des voisins.

    @output :
    Le dictionnaire des voisins est sauvegardé dans un fichier.
    """

    # Déterminez le nombre de voisins à rechercher
    k = min(len(fish_centers) - 1, len(next(iter(fish_centers.values()))))  # Nombre de poissons - 1 ou le minimum

    dico_voisin = {}
    for frame, fishes in fish_centers.items():
        frame_data = {}
        if frame not in fish_centers:
            continue  # Ignorer les frames qui ne sont pas présentes dans le dictionnaire
        positions = np.array(list(fishes.values()))  # Récupérer les positions des poissons

        if len(positions.shape) > 1:
            positions = positions.reshape(-1, 2)  # Remodeler le tableau si nécessaire

        if k > 0 and len(positions) >= k:  # Vérifiez s'il y a suffisamment d'échantillons pour trouver des voisins
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(positions)
            distances, indices = nbrs.kneighbors(positions)
            for i, (fish_id, fish_position) in enumerate(fishes.items()):
                fish_neighbors = {}
                for j, neighbor_index in enumerate(indices[i]):
                    if neighbor_index != i:  # Exclure le poisson lui-même
                        neighbor_fish_id = list(fishes.keys())[neighbor_index]
                        neighbor_distance = distances[i][j]
                        fish_neighbors[f'voisin_id {neighbor_fish_id}'] = neighbor_distance
                frame_data[f'poisson {fish_id}'] = fish_neighbors
        dico_voisin[f'frame {frame}'] = frame_data

    # Export du dictionnaire dans un fichier JSON
    with open(output, 'w') as f:
        json.dump(dico_voisin, f)


def calculate_average_distances(dico_voisin):
    """
    Calcule les distances moyennes entre chaque poisson et ses voisins.

    @input :
    - dico_voisin (dict): Le dictionnaire des voisins de chaque poisson à chaque frame.

    @output :
    - average_distances (dict): Un dictionnaire contenant les distances moyennes pour chaque poisson à chaque frame.
    """

    average_distances = {}
    for frame, frame_data in dico_voisin.items():
        frame_average_distances = {}
        for fish, neighbors in frame_data.items():
            fish_id = int(fish.split()[1])
            total_distance = sum(neighbors.values())
            average_distance = total_distance / len(neighbors)
            frame_average_distances[f'poisson {fish_id}'] = average_distance
        average_distances[frame] = frame_average_distances

    with open(neighbors_json_path, 'w') as f:
        json.dump(average_distances, f)
    return average_distances


def student_test(json_file, alpha):
    """
    Effectue un test statistique sur les distances de chaque poisson à ses voisins.

    @input :
    - json_file (str): Le chemin du fichier JSON contenant les distances des voisins.
    - alpha (float): Le niveau de signification pour le test statistique.

    @output :
    - significant_fishs (list): Une liste des poissons avec des distances significatives à leurs voisins.
    """
    significant_fishs = {}

    with open(json_file, 'r') as f:
        data = json.load(f)

    for frame in data.keys():
        distances_values = list(data[frame].values())
        for fish in data[frame].keys():
            other_distances = []
            for elem in distances_values:
                if elem != data[frame][fish]:
                    other_distances.append(elem)

            mean = np.mean(other_distances)
            std_dev = np.std(other_distances)  # ecart-type
            degrees_of_freedom = len(other_distances)
            t_statistic = (data[frame][fish] - mean) / (std_dev / np.sqrt(degrees_of_freedom + 1))
            t_critical = stats.t.ppf(1 - alpha, degrees_of_freedom)

            if abs(t_statistic) > t_critical:
                dict_frame = frame.replace("frame", "").strip()
                dict_fish = fish.replace("poisson", "").strip()
                significant_fishs[dict_frame] = dict_fish

    return significant_fishs


def plot_fish_positions(fish_centers, frame_number):
    """
    Trace les positions des poissons pour une trame donnée afin d'aider à déterminer les paramètres eps et min_samples de DBSCAN.

    @input :
    - fish_centers (dict): Dictionnaire contenant les coordonnées des poissons pour chaque trame.
    - frame_number (str): Numéro de la trame à visualiser.

    @output :
    - Visualisation des positions des poissons pour la trame spécifiée.
    """
    # Vérifier si le numéro de trame existe dans le dictionnaire
    if frame_number not in fish_centers:
        print("Le numéro de trame spécifié n'existe pas dans le dictionnaire.")
        return

    # Extraction des positions des poissons pour la trame spécifiée
    positions = [position[0] for position in fish_centers[frame_number].values()]

    # Conversion des positions en listes de coordonnées x et y séparées
    x_positions = [position[0] for position in positions]
    y_positions = [position[1] for position in positions]

    # Traçage des positions des poissons
    plt.figure(figsize=(8, 6))
    plt.scatter(x_positions, y_positions, color='blue')
    plt.title(f'Positions des poissons pour la trame {frame_number}')
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.grid(True)
    plt.show()


def cluster_fishes_DBSCAN(fish_centers, eps, min_samples):
    """
    Effectue le clustering des poissons en utilisant l'algorithme DBSCAN.

    @input :
    - fish_centers (dict): Dictionnaire contenant les coordonnées des poissons pour chaque trame.
    - eps (float): Rayon de la sphère autour d'un point pour rechercher les voisins.
    - min_samples (int): Nombre minimum de points requis pour former un cluster.

    @output :
    - all_frames_clusters (dict): Dictionnaire contenant les informations sur les clusters pour chaque trame.
    """
    all_frames_clusters = {}
    for frame_number, frame_data in fish_centers.items():
        fish_coordinates = [frame_data[str(fish_id)][0] for fish_id in frame_data]

        # Check if fish_coordinates is empty
        if not fish_coordinates:
            continue  # Skip this frame if there are no fish coordinates

        # Utilisation de DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(fish_coordinates)

        # Création des clusters
        unique_labels = set(cluster_labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Ignorer le label -1 pour le bruit
        fish_clusters = {str(i + 1): [] for i in range(num_clusters)}
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Ignorer le bruit
                fish_clusters[str(label + 1)].append((str(i + 1), fish_coordinates[i]))

        # Construction des informations sur les clusters
        cluster_info = {}
        for cluster, fishes in fish_clusters.items():
            cluster_info[cluster] = {
                "Fish Positions": [fish[1] for fish in fishes],
                "Cluster Center": [sum(pos) / len(pos) for pos in zip(*[fish[1] for fish in fishes])]
            }

        all_frames_clusters[frame_number] = {"Clusters": cluster_info}

    return all_frames_clusters


def find_optimal_cluster_number(fish_centers, max_clusters):
    """
    Utilise la méthode du coude pour déterminer le nombre optimal de clusters.

    @input :
    - fish_centers (dict): Dictionnaire contenant les coordonnées des poissons pour chaque trame.
    - max_clusters (int): Nombre maximal de clusters à tester.

    @output :
    - optimal_num_clusters (int): Nombre optimal de clusters déterminé à l'aide de la méthode du coude.
    """
    fish_coordinates = [fish_centers["0"][str(fish_id)][0] for fish_id in fish_centers["0"]]
    X = np.array(fish_coordinates)
    inertias = []
    for num_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')
    plt.show()
    optimal_num_clusters = int(input("Entrez le nombre optimal de clusters : "))
    return optimal_num_clusters


def draw_clusters_on_image(image, cluster_data):
    """
    Dessine les clusters sur une seule image.

    @input :
    - image (numpy.ndarray): Image sur laquelle dessiner les clusters.
    - cluster_data (dict): Informations sur les clusters à dessiner.

    @output :
    - frame_with_clusters (numpy.ndarray): Image avec les clusters dessinés.
    """
    frame_with_clusters = image.copy()  # Copie de l'image d'origine pour éviter de la modifier directement
    for cluster_info in cluster_data.values():  # Parcours de chaque cluster dans les données
        cluster_center = tuple(map(int, cluster_info["Cluster Center"]))  # Centre du cluster (converti en tuple)
        # Dessin du centre du cluster en vert
        cv2.circle(frame_with_clusters, cluster_center, 10, (0, 255, 0), -1)
        # Ajout du texte "Cluster" près du centre
        cv2.putText(frame_with_clusters, 'Cluster', (cluster_center[0] - 30, cluster_center[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        for fish_position in cluster_info["Fish Positions"]:  # Parcours de chaque position de poisson dans le cluster
            fish_position = tuple(map(int, fish_position))  # Convertir les coordonnées en tuple d'entiers
            # Dessin du cercle représentant la position du poisson en bleu
            cv2.circle(frame_with_clusters, fish_position, 5, (255, 0, 0), -1)
    return frame_with_clusters  # Renvoie l'image avec les clusters dessinés


def draw_clusters_on_video(video_path, clusters_info, output_path):
    """
    Dessine les clusters sur chaque trame d'une vidéo et enregistre la vidéo résultante.

    @input :
    - video_path (str): Chemin vers la vidéo d'entrée.
    - clusters_info (dict): Informations sur les clusters pour chaque trame de la vidéo.
    - output_path (str): Chemin de sortie pour enregistrer la vidéo avec les clusters dessinés.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if str(frame_number) in clusters_info:
            frame_clusters = clusters_info[str(frame_number)]["Clusters"]
            frame_with_clusters = draw_clusters_on_image(frame, frame_clusters)
            out.write(frame_with_clusters)
            cv2.imshow('Frame', frame_with_clusters)
            print(f"Processed frame {frame_num}/{total_frames}")
        else:
            out.write(frame)
            cv2.imshow('Frame', frame)
            print(f"Processed frame {frame_num}/{total_frames} (No clusters)")

        frame_num += 1  # Increment frame number

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()  # Fermer toutes les fenêtres OpenCV après la boucle


def find_break_points(x_values, y_values):
    derivatives = []
    break_points = []

    # Calculate derivatives using finite differences
    for i in range(1, len(y_values)):
        derivative = (y_values[i] - y_values[i - 1]) / (x_values[i] - x_values[i - 1])
        derivatives.append(derivative)

    # Find x values where derivative is above 10
    for i, derivative in enumerate(derivatives):
        if abs(derivative) > 10:
            break_points.append(x_values[i + 1])  # Shift by 1 to match with x_values index

    # Find position where derivative is minimum among break points
    min_derivative_pos = None
    min_derivative = float('inf')
    for i, point in enumerate(break_points):
        if i == 0:
            continue
        if derivatives[i] < min_derivative:
            min_derivative = derivatives[i]
            min_derivative_pos = point

    return break_points, derivatives, min_derivative_pos


def determine_num_clusters_from_dict(fish_centers, max_clusters):
    fish_coordinates = [fish_centers["0"][str(fish_id)][0] for fish_id in fish_centers["0"]]
    X = np.array(fish_coordinates)
    inertias = []
    for num_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')

    # Find break points on the elbow curve
    break_points, derivatives, min_derivative_pos = find_break_points(range(1, max_clusters + 1), inertias)
    print("Break points:", break_points)
    print("Derivatives:", derivatives)
    print("Min derivative position:", min_derivative_pos)
    for point in break_points:
        plt.axvline(x=point, color='r', linestyle='--', linewidth=0.5)  # Draw red vertical line for each break point

    # plt.show()
    optimal_num_clusters = min_derivative_pos
    print(optimal_num_clusters)
    return optimal_num_clusters


def cluster_fishes_KMEANS(fish_centers, num_clusters):
    """
    Effectue le clustering des poissons en utilisant l'algorithme K-Means.

    @input :
    - fish_centers (dict): Dictionnaire contenant les coordonnées des poissons pour chaque trame.
    - num_clusters (int): Nombre de clusters à former.

    @output :
    - all_frames_clusters (dict): Dictionnaire contenant les informations sur les clusters pour chaque trame.
    """
    all_frames_clusters = {}
    for frame_number, frame_data in fish_centers.items():
        fish_coordinates = [frame_data[str(fish_id)][0] for fish_id in frame_data]

        # Utilisation de l'initialisation K-Means++
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++')
        kmeans.fit(fish_coordinates)
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_

        fish_clusters = {str(i + 1): [] for i in range(num_clusters)}
        for i, label in enumerate(cluster_labels):
            fish_clusters[str(label + 1)].append((str(i + 1), fish_coordinates[i]))

        cluster_info = {}
        for cluster, fishes in fish_clusters.items():
            cluster_info[cluster] = {
                "Fish Positions": [fish[1] for fish in fishes],
                "Cluster Center": [float(coord) for coord in cluster_centers[int(cluster) - 1]]
            }

        all_frames_clusters[frame_number] = {"Clusters": cluster_info}

    return all_frames_clusters


def find_optimal_cluster_number(fish_centers, max_clusters):
    """
    Utilise la méthode du coude pour déterminer le nombre optimal de clusters.

    @input :
    - fish_centers (dict): Dictionnaire contenant les coordonnées des poissons pour chaque trame.
    - max_clusters (int): Nombre maximal de clusters à tester.

    @output :
    - optimal_num_clusters (int): Nombre optimal de clusters déterminé à l'aide de la méthode du coude.
    """
    fish_coordinates = [fish_centers["0"][str(fish_id)][0] for fish_id in fish_centers["0"]]
    X = np.array(fish_coordinates)
    inertias = []
    # Déterminer le nombre maximal de clusters en fonction du nombre d'échantillons
    max_clusters = min(max_clusters, len(X))
    for num_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    # Plot the elbow curve
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')

    # Find the optimal number of clusters using the derivative
    break_points, derivatives, min_derivative_pos = find_break_points(range(1, max_clusters + 1), inertias)
    optimal_num_clusters = min_derivative_pos

    if optimal_num_clusters is None:
        # Si optimal_num_clusters est None, définissez-le comme un tiers du nombre total de poissons
        optimal_num_clusters = len(fish_centers["0"]) // 2

    print(f"Nombre optimal de clusters déterminé : {optimal_num_clusters}")
    return optimal_num_clusters


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    openvino_model_path = os.path.join(current_dir, 'runs/detect/train12/weights/best_openvino_model')
    fps_file = os.path.join(current_dir, 'fps/')
    video_dir = os.path.join(current_dir, 'videos/')
    detection_dir = os.path.join(current_dir, 'detection/')
    center_video_dir = os.path.join(current_dir, 'center_video/')
    group_center_dir = os.path.join(current_dir, 'group_center/')
    heatmap_dir = os.path.join(current_dir, 'heatmaps/')
    tracking_dir = os.path.join(current_dir, 'tracking/')
    json_dir = os.path.join(current_dir, 'json/')
    isolation_fish_dir = os.path.join(current_dir, 'isolation_fish/')
    speed_dir = os.path.join(current_dir, 'speed/')
    DBSCAN_dir = os.path.join(current_dir, 'cluster_DBSCAN/')
    KMEANS_dir = os.path.join(current_dir, 'clusters_KMEANS/')

    # Création des dossiers s'ils n'existent pas
    for directory in [detection_dir, center_video_dir, group_center_dir, heatmap_dir, tracking_dir, fps_file, json_dir,
                      isolation_fish_dir, speed_dir, KMEANS_dir, DBSCAN_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    index = 16
    for video_file in os.listdir(video_dir):
        print(f"\n - - - Video {index} : {video_file} - - - ")
        video_path = os.path.join(video_dir, video_file)
        video_detection_path = os.path.join(detection_dir, f"video_detection_{index}.mp4")
        fish_center_video_path = os.path.join(center_video_dir, f"fish_center_video_{index}.mp4")
        group_centers_video_path = os.path.join(group_center_dir, f"group_centers_video_{index}.mp4")
        tracking_video_path = os.path.join(tracking_dir, f"tracking_video_{index}.mp4")
        farthest_fish_video_path = os.path.join(isolation_fish_dir, f"farthest_fish_video_{index}.mp4")
        isolation_video_path = os.path.join(isolation_fish_dir, f"neighbors_isolation{index}.mp4")
        DBSCAN_video_path = os.path.join(DBSCAN_dir, f'clusters_DBSCAN_video{index}.mp4')
        KMEANS_video_path = os.path.join(KMEANS_dir, f'clusters_KMEANS_video{index}.mp4')

        fps_path = os.path.join(fps_file, f"fps_{index}.txt")

        fish_centers_json_path = os.path.join(json_dir, f"fish_centers_{index}.json")
        group_centers_json_path = os.path.join(json_dir, f"group_centers_{index}.json")
        neighbors_json_path = os.path.join(json_dir, f"neighbors_average_distance_{index}.json")
        neigh_json_path = os.path.join(json_dir, f"neighbors{index}.json")
        count_farthest_fish_path = os.path.join(json_dir, f"ct_farthest_fish{index}.json")

        fish_speeds_plot_path = os.path.join(speed_dir, f"courbes_vitesse_{index}.jpg")
        heatmap_image_path = os.path.join(heatmap_dir, f"heatmap_image_{index}.png")

        # Demander à l'utilisateur ce qu'il veut faire
        print(" -- MENU POUR TOUTES LES VIDEOS --\n")
        print('1- Entraînement du modèle')
        print('2- Detection des positions des poissons sur une vidéo')
        print('3- Affichage centre poisson')
        print('4- Affichage des centres des groupes des poissons sur une vidéo')
        print('5- Heatmap sur une vidéo')
        print('6- Tracking sur une vidéo')
        print('7- Trouver le poisson le plus éloigné')
        print('8- Trouver la  vitesse de chaque poisson')
        print('9- Proches voisins')
        print('10- Clusters avec méthode DBSCAN sur une vidéo')
        print('11- Clusters avec méthode KMEANS sur une vidéo')

        choix = input("Que voulez-vous faire ?\n ")

        if choix == "1":  # Entraînement du modèle
            model_training(current_dir)

        elif choix == "2":  # Detection des positions des poissons sur une vidéo
            # Importation du modèle enregistré
            model = YOLO(openvino_model_path)

            # Detection des poissons sur une vidéo
            output_video_path = video_detection_path
            dico_position, fps = video_detection(video_path, output_video_path, model)

            # Création d'un dictionnaire avec les positions des poissons par frame
            fish_centers = fish_position_dico(dico_position)

            with open(fish_centers_json_path, 'w') as f:
                json.dump(fish_centers, f)

            with open(fps_path, 'w') as f:
                f.write(str(fps))

        elif choix == "3":  # Affichage center poisson
            with open(fish_centers_json_path, 'r') as f:
                fish_centers = json.load(f)

            display_fish_centers(fish_centers, video_path, fish_center_video_path, fps_file)

        elif choix == "4":  # Affichage des centres des groupes des poissons sur une vidéo
            with open(fish_centers_json_path, 'r') as f:
                fish_centers = json.load(f)

            video_path = video_detection_path
            output_video = group_centers_video_path

            display_group_center(fish_centers, video_path, output_video)

        elif choix == "5":  # Heatmap sur une vidéo

            model = YOLO(openvino_model_path)
            output_image = heatmap_image_path

            # Appel de la fonction pour générer l'image de la carte de chaleur
            generate_heatmap(model, video_path, output_image)

        elif choix == "6":  # Tracking sur une vidéo
            model = YOLO(openvino_model_path)
            output_video = tracking_video_path

            tracking(model, video_path, output_video)

        elif choix == "7":  # Trouver le poisson le plus éloigné
            # Charger les données des fichiers JSON
            with open(fish_centers_json_path, 'r') as f:
                fish_centers = json.load(f)

            with open(group_centers_json_path, 'r') as f:
                group_centers = json.load(f)

            # Appeler la fonction pour trouver le poisson le plus éloigné à chaque frame
            farthest_fish_per_frame = find_farthest_fish(fish_centers, group_centers)

            # Affichage des résultats
            for frame_num, farthest_fish_index in farthest_fish_per_frame.items():
                print(f"Frame {frame_num}: Poisson le plus éloigné - Poisson_{farthest_fish_index}")

            print("Nombre de fois que chaque poisson est le plus éloigné :")
            fish_counts = count_farthest_fish(farthest_fish_per_frame, count_farthest_fish_path)

            for fish_id, count in fish_counts.items():
                print(f"Poisson {fish_id} : {count} fois")

            output_video = isolation_video_path
            draw_farthest_fish(fish_centers, farthest_fish_per_frame, video_path, output_video)

        elif choix == "8":  # Trouver la  vitesse de chaque poisson
            # Calcul de la vitesse des poissons
            with open(fish_centers_json_path, 'r') as f:
                fish_centers = json.load(f)

            with open(fps_path, 'r') as f:
                fps = float(f.read())

            fish_speeds = calculate_fish_speeds(fish_centers, fps)
            output_dir = os.path.dirname(fish_speeds_plot_path)

            # Créer le répertoire s'il n'existe pas
            os.makedirs(output_dir, exist_ok=True)
            plot_fish_speeds(fish_speeds, fish_speeds_plot_path)

        elif choix == "9":  # Proches voisins Student Test

            output_video_path = isolation_video_path
            json_file = neighbors_json_path

            # Charger le fichier JSON
            with open(fish_centers_json_path, "r") as f:
                fish_centers = json.load(f)

            create_dict_neighbors(fish_centers, neigh_json_path)

            with open(neigh_json_path, "r") as f:
                neighbors_dict = json.load(f)

            calculate_average_distances(neighbors_dict)
            isolated_poissons_per_frame = student_test(json_file, alpha=0.005)

            with open(fish_centers_json_path, "r") as f:
                fish_centers = json.load(f)
            draw_farthest_fish(fish_centers, isolated_poissons_per_frame, video_path, output_video_path)


        elif choix == "10":  # Cluster, Méthode avec DBSCAN

            with open(fish_centers_json_path, 'r') as f:
                fish_centers = json.load(f)

            eps = 200  # Distance maximale entre deux échantillons pour être dans le même cluster
            min_samples = 1  # Nombre minimum d'échantillons dans un cluster
            # plot_fish_positions(fish_centers, "0")
            clusters_info = cluster_fishes_DBSCAN(fish_centers, eps, min_samples)

            # Chemin de la vidéo d'entrée
            video_path = video_path
            output_video = DBSCAN_video_path

            draw_clusters_on_video(video_path, clusters_info, output_video)

        elif choix == "11":
            # Charger les données des poissons depuis le fichier JSON
            with open(fish_centers_json_path, 'r') as f:
                fish_centers = json.load(f)

            if not fish_centers:
                print("Erreur: Aucune donnée de poissons disponible.")
            else:
                # Vérifier si au moins une trame est disponible
                if len(fish_centers) > 0:
                    # Trouver le nombre optimal de clusters pour la première frame de la vidéo
                    max_clusters = len(fish_centers["0"].values())
                    num_clusters = find_optimal_cluster_number(fish_centers, max_clusters)

                    # Appliquer l'algorithme K-Means pour le clustering des poissons
                    clusters_info = cluster_fishes_KMEANS(fish_centers, num_clusters)

                    # Créer la vidéo avec les clusters dessinés
                    video_path = video_path
                    output_video = KMEANS_video_path
                    draw_clusters_on_video(video_path, clusters_info, output_video)
                else:
                    print("Erreur: Aucune trame de données de poissons disponible.")


        else:
            print("Choix invalide. Veuillez recommencer")

    index += 1


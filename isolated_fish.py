
# Importation des bibliothèques

import cv2
import math
import json



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
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

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


def calculate_distance(point1, point2):
    """
    Calcule la distance euclidienne entre deux points.

    @input :
    - point1 (tuple): Les coordonnées du premier point.
    - point2 (tuple): Les coordonnées du deuxième point.

    @output :
    - La distance entre les deux points.
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


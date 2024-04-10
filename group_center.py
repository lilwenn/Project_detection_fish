
# Impotation des bibliothèques


import cv2
import json

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


def display_group_center(fish_centers, video_path, output_video_path,group_centers_json_path, fps_file):
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
    with open(fps_file, 'r') as f:
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

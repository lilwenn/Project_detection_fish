
# Impotation des bibliothèques

import cv2

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
            x1, y1, x2, y2 = positions[0],positions[1],positions[2], positions[3]
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


def display_fish_centers(fish_centers, video_path, output_video_path, fps_file):
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

    with open(fps_file, 'r') as f:
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


# Impotation des bibliothèques

import matplotlib.pyplot as plt
import math
import numpy as np


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
                    distance = math.sqrt((center[0][0] - prev_center[0][0])**2 + (center[0][1] - prev_center[0][1])**2)
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


#Importation des fonctions

from model_train import model_training
from position_detection import video_detection, fish_position_dico, display_fish_centers
from group_center import display_group_center
from heatmap import generate_heatmap
from isolated_fish import find_farthest_fish, count_farthest_fish, draw_farthest_fish
from speed import calculate_fish_speeds, plot_fish_speeds
from significant_isolated_fish import create_dict_neighbors, calculate_average_distances, student_test, write_significant_fish_to_excel, calculate_isolated_fish_percentage
from cluster import  draw_clusters_on_video,cluster_fishes_DBSCAN, write_cluster_info_to_excel, color_excel_cells, isolation_alert_threshold, calculate_fish_distribution, plot_single_fish_clusters_pie_chart, plot_fish_distribution_pie_chart
from analyse import read_excel_data, find_frames_with_single_fish_and_significant_frames, excel_isolated_fish_creation
# Impotation des bibliothèques

from ultralytics import YOLO
import json
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    openvino_model_path = os.path.join(current_dir, 'runs/detect/train12/weights/best_openvino_model')
    fps_path = os.path.join(current_dir, 'fps/')
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
    excel_dir = os.path.join(current_dir, 'excel/')

    x_dim_bath = 34.5
    y_dim_bath = 17.8
    bath_area = x_dim_bath * y_dim_bath
    print (bath_area)
    # Création des dossiers s'ils n'existent pas
    for directory in [detection_dir, center_video_dir, excel_dir, group_center_dir, heatmap_dir, tracking_dir, fps_path, json_dir, isolation_fish_dir, speed_dir, KMEANS_dir, DBSCAN_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    index = 0
    for video_file in os.listdir(video_dir):

        print(f"\n - - - Video {index} : {video_file} - - - ")
        video_path = os.path.join(video_dir, video_file)
        video_detection_path = os.path.join(detection_dir, f"video_detection_{index}.mp4")
        fish_center_video_path = os.path.join(center_video_dir, f"fish_center_video_{index}.mp4")
        group_centers_video_path = os.path.join(group_center_dir, f"group_centers_video_{index}.mp4")
        tracking_video_path = os.path.join(tracking_dir, f"tracking_video_{index}.mp4")
        farthest_fish_video_path = os.path.join(isolation_fish_dir, f"farthest_fish_video_{index}.mp4")
        nei_isolation_video_path = os.path.join(isolation_fish_dir, f"neighbors_isolation{index}.mp4")
        dist_isolation_video_path = os.path.join(isolation_fish_dir, f"distance_isolation{index}.mp4")
        DBSCAN_video_path = os.path.join(DBSCAN_dir, f'clusters_DBSCAN_video{index}.mp4')
        KMEANS_video_path = os.path.join(KMEANS_dir, f'clusters_KMEANS_video{index}.mp4')

        fps_file = os.path.join(fps_path, f"fps_{index}.txt")

        fish_centers_json_path = os.path.join(json_dir, f"fish_centers_{index}.json")
        group_centers_json_path = os.path.join(json_dir, f"group_centers_{index}.json")
        neighbors_json_path = os.path.join(json_dir, f"neighbors_average_distance_{index}.json")
        neigh_json_path = os.path.join(json_dir, f"neighbors{index}.json")
        count_farthest_fish_path = os.path.join(json_dir, f"ct_farthest_fish{index}.json")

        significant_fishs_path = os.path.join(excel_dir, f"significant_fishs{index}.xlsx")
        clusters_info_path = os.path.join(excel_dir, f"clusters_info{index}.xlsx")
        isolated_exl_file = os.path.join(excel_dir, f"isolated_fish{index}.xlsx")

        fish_speeds_plot_path = os.path.join(speed_dir, f"courbes_vitesse_{index}.jpg")
        heatmap_image_path = os.path.join(heatmap_dir, f"heatmap_image_{index}.png")

        pie1_image_path = os.path.join(current_dir, f"pie1_clusters{index}.png")
        pie2_image_path = os.path.join(current_dir, f"pie2_clusters{index}.png")
    
        # Demander à l'utilisateur ce qu'il veut faire
        print(" -- MENU POUR TOUTES LES VIDEOS --\n")
        print('1- Entraînement du modèle')
        print('2- Detection des positions des poissons sur une vidéo + affichage')
        print('3- Affichage des centres des groupes des poissons sur une vidéo')
        print('4- Heatmap sur une vidéo')
        print('5- Trouver le poisson le plus éloigné')
        print('6- Trouver la  vitesse de chaque poisson')
        print('7- Isolation avec proches voisins sur une vidéo')
        print('8- Affichage des clusters avec méthode DBSCAN sur une vidéo')

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

            with open(fps_file, 'w') as f:
                f.write(str(fps))
            # Affichage centre poisson

            with open(fish_centers_json_path, 'r') as f:
                fish_centers = json.load(f)

            display_fish_centers(fish_centers, video_path, fish_center_video_path, fps_file)



        elif choix == "3":  # Affichage des centres des groupes des poissons sur une vidéo

            with open(fish_centers_json_path, 'r') as f:
                fish_centers = json.load(f)

            video_path = video_detection_path
            output_video = group_centers_video_path

            display_group_center(fish_centers, video_path, output_video, group_centers_json_path, fps_file)

        elif choix == "4":  # Heatmap sur une vidéo

            model = YOLO(openvino_model_path)
            output_image = heatmap_image_path

            # Appel de la fonction pour générer l'image de la carte de chaleur
            generate_heatmap(model, video_path, output_image)


        elif choix == "5":  # Trouver le poisson le plus éloigné

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

            video_path = video_detection_path
            output_video = dist_isolation_video_path

            draw_farthest_fish(fish_centers, farthest_fish_per_frame, video_path, output_video)


        elif choix == "6":  # Trouver la  vitesse de chaque poisson
            # Calcul de la vitesse des poissons
            with open(fish_centers_json_path, 'r') as f:
                fish_centers = json.load(f)

            with open(fps_file, 'r') as f:
                fps = float(f.read())
            fish_speeds = calculate_fish_speeds(fish_centers, fps)
            # print(fish_speeds)
            plot_fish_speeds(fish_speeds, fish_speeds_plot_path)


        elif choix == "7":  # Isolation Proches voisins

            output_video_path = nei_isolation_video_path
            json_file = neighbors_json_path

            # Charger le fichier JSON
            with open(fish_centers_json_path, "r") as f:
                fish_centers = json.load(f)

            create_dict_neighbors(fish_centers, neigh_json_path)

            with open(neigh_json_path, "r") as f:
                neighbors_dict = json.load(f)

            calculate_average_distances(neighbors_dict, neighbors_json_path)
            # Appel de la fonction student_test

            # Appel de la fonction student_test
            significant_fishs, total_frames = student_test(json_file, alpha=0.0005)

            # Appel de la fonction pour calculer le pourcentage de poissons isolés
            calculate_isolated_fish_percentage(significant_fishs, total_frames)

            # Ensuite, vous pouvez appeler la fonction pour écrire les poissons significatifs dans un fichier Excel
            write_significant_fish_to_excel(significant_fishs, significant_fishs_path)

            with open(fish_centers_json_path, "r") as f:
                fish_centers = json.load(f)
            draw_farthest_fish(fish_centers, significant_fishs, video_path, output_video_path)

            #Excel
    
            write_significant_fish_to_excel(significant_fishs, significant_fishs_path)


        elif choix == "8": # DBSCAN
            with open(fish_centers_json_path, 'r') as f:
                fish_centers = json.load(f)

            # Méthode avec DBSCAN
            eps = bath_area / 3  # Distance maximale entre deux échantillons pour être dans le même cluster
            min_samples = 1  # Nombre minimum d'échantillons dans un cluster
            # plot_fish_positions(fish_centers, "0")
            clusters_info = cluster_fishes_DBSCAN(fish_centers, eps, min_samples)

            # Chemin de la vidéo d'entrée
            video_path = video_path
            output_video = DBSCAN_video_path

            draw_clusters_on_video(video_path, clusters_info, output_video)


            # EXCEL
            excel_file_path = clusters_info_path
            write_cluster_info_to_excel(clusters_info, excel_file_path)
            color_excel_cells(excel_file_path)
            
            
            #ALERTE isolement TAC
            alert_threshold = 20        
            isolation_alert_threshold(excel_file_path, alert_threshold) 
            
            # Calculer la distribution des poissons
            fish_distribution = calculate_fish_distribution(excel_file_path)

            # Afficher le diagramme circulaire de répartition des clusters par nombre de poissons
            plot_single_fish_clusters_pie_chart(excel_file_path, pie1_image_path)

            # Afficher le diagramme circulaire de répartition des clusters de poissons
            plot_fish_distribution_pie_chart(fish_distribution, pie2_image_path)
        

        else:
            print("Choix invalide. Veuillez recommencer")

        if os.path.isfile(significant_fishs_path) and os.path.isfile(clusters_info_path):
            significant_df, clusters_df = read_excel_data(significant_fishs_path, clusters_info_path)
            single_fish_frames, significant_frames, common_frames = find_frames_with_single_fish_and_significant_frames(significant_df, clusters_df)
            excel_isolated_fish_creation(single_fish_frames, significant_frames, common_frames, isolated_exl_file )
        


    index += 1

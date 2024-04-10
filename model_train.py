
import os
from ultralytics import YOLO
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
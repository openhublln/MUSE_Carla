import os
import time
from queue import Empty

def create_scene_folders(scene_id, sensor_names, base_save_path):
    """ Crée les dossiers pour chaque scène d'après les noms des capteurs de la config """
    scene_path = os.path.join(base_save_path, f"scene_{scene_id}")
    os.makedirs(scene_path, exist_ok=True)

    for folder in sensor_names:
        os.makedirs(os.path.join(scene_path, folder), exist_ok=True)

    return scene_path

def run_simulation(scene_id, world, vehicle, sensor_list, sensor_queue, ticks_per_scene):
    """ Exécute une simulation en s'assurant que chaque tick contient bien une donnée pour chaque capteur. """

    print(f"Simulation {scene_id} démarrée...")

    try:
        for tick in range(ticks_per_scene):  
            world.tick()
            snapshot = world.get_snapshot()
            world_timestamp = int(snapshot.timestamp.elapsed_seconds * 1e3)  # Timestamp en millisecondes
            w_frame = snapshot.frame

            print(f"Scene {scene_id} - Tick {tick+1}/{ticks_per_scene} - World frame: {w_frame} - Timestamp: {world_timestamp}")

            # Dictionnaire pour stocker les données de chaque capteur
            received_sensors = {}

            # Attendre les données de chaque capteur
            while len(received_sensors) < len(sensor_list):
                try:
                    s_timestamp, s_name = sensor_queue.get(True, 1.0)
                    received_sensors[s_name] = s_timestamp
                except Empty:
                    print("    Données de capteur manquées !")
                    break  # On passe au tick suivant même s'il manque des capteurs

            # Afficher toutes les données reçues pour ce tick
            for sensor_name, sensor_timestamp in received_sensors.items():
                print(f"    Sensor Timestamp: {world_timestamp}   Sensor: {sensor_name}")

    except Exception as e:
        print(f"Erreur pendant la simulation: {e}")

    finally:
        print(f"Nettoyage des acteurs pour la scène {scene_id}...")
        for sensor in sensor_list:
            if sensor.is_alive:
                sensor.destroy()
        if vehicle is not None and vehicle.is_alive:
            vehicle.destroy()
        time.sleep(1)
        print(f"Scène {scene_id} terminée.")
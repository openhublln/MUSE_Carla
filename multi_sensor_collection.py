import json
import datetime
from queue import Queue, Empty
import os
import carla
import random
import numpy as np
import time
import yaml
import math
from generate_bbox_annotations import process_scene

def create_scene_folders(scene_id, sensor_names, base_save_path):
    """ Crée les dossiers pour chaque scène d'après les noms des capteurs de la config """
    scene_path = os.path.join(base_save_path, f"scene_{scene_id}")
    os.makedirs(scene_path, exist_ok=True)

    for folder in sensor_names:
        os.makedirs(os.path.join(scene_path, folder), exist_ok=True)

    return scene_path

def calculate_radar_intensity(depth):
    """Calcule l'intensité du signal radar."""
    rcs = 10          # Section efficace radar moyenne (m²)
    noise_floor = 1e-9
    ref_distance = 10 # Distance de référence (m)
    intensity = (ref_distance / depth) ** 4 * rcs
    noise = np.random.normal(0, noise_floor)
    return max(intensity + noise, 0)

def sensor_callback(sensor_data, sensor_queue, sensor_name, save_path, world=None, ego_vehicle=None):
    """ Callback pour traiter et enregistrer les données des capteurs """
    try:
        timestamp = int(sensor_data.timestamp * 1e3)
        sensor_folder = os.path.join(save_path, sensor_name)

        if isinstance(sensor_data, carla.Image):
            img_filename = f"{timestamp}.png"
            img_path = os.path.join(sensor_folder, img_filename)
            
            if "semantic" in sensor_name:
                sensor_data.save_to_disk(img_path, carla.ColorConverter.CityScapesPalette)
            elif "instance" in sensor_name:
                sensor_data.save_to_disk(img_path)
            else:
                sensor_data.save_to_disk(img_path)  
        
        elif isinstance(sensor_data, carla.SemanticLidarMeasurement):
            lidar_filename = os.path.join(sensor_folder, f"{timestamp}.ply")
            # Save as PLY file for compatibility with visualization tools
            sensor_data.save_to_disk(lidar_filename)
            
            # Also save as NPY for structured data access
            npy_filename = os.path.join(sensor_folder, f"{timestamp}.npy")
            points = np.frombuffer(sensor_data.raw_data, dtype=np.dtype([
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('cos_inc_angle', np.float32),
                ('object_idx', np.uint32), ('semantic_tag', np.uint32)
            ]))
            np.save(npy_filename, points)
        
        elif isinstance(sensor_data, carla.LidarMeasurement):
            lidar_filename = os.path.join(sensor_folder, f"{timestamp}.npy")
            points = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (-1, 4))  
            np.save(lidar_filename, points)
        
        elif isinstance(sensor_data, carla.RadarMeasurement):
            points = []
            for detect in sensor_data:
                azi = math.degrees(detect.azimuth)
                alt = math.degrees(detect.altitude)
                intensity = calculate_radar_intensity(detect.depth)
                points.append([detect.depth, alt, azi, detect.velocity, intensity])
            points = np.array(points, dtype=np.float32)
            np.save(os.path.join(sensor_folder, f"{timestamp}.npy"), points)
        
        sensor_queue.put((timestamp, sensor_name))
        
    except Exception as e:
        print(f"Error in sensor callback: {e}")
        sensor_queue.put((0, sensor_name))  # Put dummy data to avoid blocking

def run_simulation(scene_id, world, vehicle, sensor_list, sensor_queue, ticks_per_scene, weather):
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

def clean_scene_data(scene_path, sensor_names):
    """
    Nettoie le jeu de données d'une scène en supprimant les fichiers dont le timestamp 
    n'est pas présent dans tous les dossiers de capteurs.
    """
    # Construire un dictionnaire des ensembles de timestamps pour chaque capteur
    ts_dict = {}
    for sensor in sensor_names:
        sensor_folder = os.path.join(scene_path, sensor)
        if not os.path.isdir(sensor_folder):
            continue
        
        # Récupérer seulement les fichiers PNG et NPY qui ne sont pas des instances
        files = []
        for f in os.listdir(sensor_folder):
            if f.endswith('.png'):
                files.append(f)
            elif f.endswith('.npy') and not f.endswith('_instances.npy'):
                files.append(f)
        
        # Extraire les timestamps (sans l'extension)
        ts_set = set(os.path.splitext(f)[0] for f in files)
        ts_dict[sensor] = ts_set

    if not ts_dict:
        return

    # Calculer l'intersection de tous les timestamps
    common_ts = set.intersection(*ts_dict.values())

    # Pour chaque capteur, supprimer les fichiers non communs
    for sensor, timestamps in ts_dict.items():
        sensor_folder = os.path.join(scene_path, sensor)
        for file_name in os.listdir(sensor_folder):
            base_name = os.path.splitext(file_name)[0]
            
            if base_name not in common_ts:
                file_path = os.path.join(sensor_folder, file_name)
                try:
                    os.remove(file_path)
                    print(f"Deleted non-common file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

def setup_traffic(client, world, traffic_config):
    """Configure and spawn traffic (vehicles and pedestrians)"""
    try:
        # Get blueprints for vehicles and pedestrians
        blueprints = world.get_blueprint_library().filter('vehicle.*')
        if traffic_config.get("safe_spawn", True):
            blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']
        blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')

        # Get spawn points
        spawn_points = world.get_map().get_spawn_points()
        num_spawn_points = len(spawn_points)
        num_vehicles = min(traffic_config.get("num_vehicles", 30), num_spawn_points)

        # Spawn vehicles
        batch = []
        vehicle_list = []
        for n, transform in enumerate(random.sample(spawn_points, num_vehicles)):
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(carla.command.SpawnActor(blueprint, transform))

        # Apply batch spawn
        for response in client.apply_batch_sync(batch, True):
            if response.error:
                print(f"Error spawning vehicle: {response.error}")
            else:
                vehicle_list.append(response.actor_id)

        # Set up traffic manager
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.global_percentage_speed_difference(30.0)

        # Set vehicles to autopilot
        vehicles = world.get_actors(vehicle_list)
        for vehicle in vehicles:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            if traffic_config.get("car_lights_on", False):
                vehicle.set_light_state(carla.VehicleLightState.All)

        print(f"Spawned {len(vehicle_list)} vehicles")
        return vehicle_list

    except Exception as e:
        print(f"Error setting up traffic: {e}")
        return []

def process_sensor_config(sensors_config):
    """Process sensor configuration and automatically add instance segmentation cameras"""
    processed_config = []
    
    for sensor in sensors_config:
        # Add the original sensor configuration
        processed_config.append(sensor)
        
        # If it's a camera with collect_bbox enabled, add corresponding instance segmentation camera
        if (sensor.get("type") == "camera" and 
            sensor.get("blueprint") == "sensor.camera.rgb" and 
            sensor.get("collect_bbox", False)):
            
            # Create instance segmentation camera config
            instance_camera = sensor.copy()
            instance_camera["name"] = f"instance_{sensor['name']}"
            instance_camera["blueprint"] = "sensor.camera.instance_segmentation"
            instance_camera.pop("collect_bbox", None)  # Remove collect_bbox flag
            
            processed_config.append(instance_camera)
    
    return processed_config

def main():
    """ Initialise Carla, configure les paramètres et lance les simulations """
    try:
        # Charger la configuration depuis le fichier YAML
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Process sensor configuration to add instance cameras
        config["sensors"] = process_sensor_config(config["sensors"])
        
        sim_config = config["simulation"]
        sensors_config = config["sensors"]
        weather_config = config.get("weather", {})
        traffic_config = config.get("traffic", {})

        num_scenes = sim_config["num_scenes"]
        ticks_per_scene = sim_config["ticks_per_scene"]
        base_save_path = sim_config["base_save_path"]

        # Set random seed if specified
        if "seed" in traffic_config and traffic_config["seed"] is not None:
            random.seed(traffic_config["seed"])

        # Connexion au client
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)  # Increased timeout

        # Récupération du monde et configuration du mode synchrone
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20Hz
        world.apply_settings(settings)

        # Configurer la météo en utilisant un preset s'il est défini
        preset = weather_config.get("preset", None)
        if preset:
            try:
                weather = getattr(carla.WeatherParameters, preset)
            except AttributeError:
                print(f"Preset '{preset}' non trouvé. Utilisation de ClearNoon par défaut.")
                weather = carla.WeatherParameters.ClearNoon
        else:
            weather = carla.WeatherParameters(
                cloudiness = weather_config.get("cloudiness", 0.0),
                precipitation = weather_config.get("precipitation", 0.0),
                precipitation_deposits = weather_config.get("precipitation_deposits", 0.0),
                wind_intensity = weather_config.get("wind_intensity", 0.0),
                fog_density = weather_config.get("fog_density", 0.0),
                fog_distance = weather_config.get("fog_distance", 0.0),
                fog_falloff = weather_config.get("fog_falloff", 0.0),
                wetness = weather_config.get("wetness", 0.0),
                sun_azimuth_angle = weather_config.get("sun_azimuth_angle", 0.0),
                sun_altitude_angle = weather_config.get("sun_altitude_angle", 90.0)
            )
        world.set_weather(weather)
        print("Météo initiale :", world.get_weather())

        # Setup traffic before starting sensor collection
        print("Setting up traffic...")
        vehicle_list = setup_traffic(client, world, traffic_config)
        
        # Let the traffic settle for a few seconds
        print("Letting traffic settle...")
        for _ in range(50):  # 2.5 seconds at 20Hz
            world.tick()

        # Configuration du traffic manager
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        scene_paths = []  # Liste pour stocker les dossiers de scènes traitées

        for scene_id in range(1, num_scenes + 1):
            # Création des dossiers pour la scène via la fonction adaptée
            sensor_names = [s["name"] for s in sensors_config]
            save_path = create_scene_folders(scene_id, sensor_names, base_save_path)
            scene_paths.append(save_path)

            # Création de la file d'attente des capteurs et récupération du blueprint_library
            sensor_queue = Queue()
            blueprint_library = world.get_blueprint_library()

            # Spawn du véhicule
            spawn_points = world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)
            bp = blueprint_library.find('vehicle.lincoln.mkz')
            vehicle = world.spawn_actor(bp, spawn_point)
            vehicle.set_autopilot(True, traffic_manager.get_port())

            print("Véhicule spawné, attente de stabilisation...")
            for _ in range(20):  # Increased stabilization ticks
                try:
                    world.tick()
                except:
                    time.sleep(0.1)
            
            try:
                vehicle.set_autopilot(True, traffic_manager.get_port())
            except Exception as e:
                print(f"Warning: Could not enable autopilot: {e}")
                
            print("Véhicule prêt, démarrage des capteurs.")

            sensor_list = []
            # Création des capteurs depuis la configuration
            for sensor in sensors_config:
                bp_sensor = blueprint_library.find(sensor["blueprint"])
                for attr, value in sensor["attributes"].items():
                    bp_sensor.set_attribute(attr, value)
                loc = sensor["transform"]["location"]
                rot = sensor["transform"]["rotation"]
                transform = carla.Transform(
                    carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
                    carla.Rotation(pitch=rot.get("pitch", 0), yaw=rot["yaw"], roll=rot.get("roll", 0))
                )
                actor = world.spawn_actor(bp_sensor, transform, attach_to=vehicle)
                sensor_list.append(actor)
                # Passer world et vehicle au callback
                actor.listen(lambda data, q=sensor_queue, name=sensor["name"], 
                           path=save_path, w=world, v=vehicle:
                             sensor_callback(data, q, name, path, w, v))
            
            # Lancer la simulation en passant ticks_per_scene depuis la config
            run_simulation(scene_id, world, vehicle, sensor_list, sensor_queue, ticks_per_scene, weather)
        print("Toutes les scènes ont été simulées !")
        
        # Nettoyer toutes les scènes après la fin des simulations
        for path in scene_paths:
            print("Nettoyage du dataset pour la scène", path)
            clean_scene_data(path, sensor_names)
        print("Toutes les scènes ont été nettoyées avec succès !")
        
        # Generate bounding box annotations
        print("\nGenerating bounding box annotations...")
        for path in scene_paths:
            process_scene(path)
        print("Bounding box annotation generation complete!")
    
    except KeyboardInterrupt:
        print(" - Interrompu par l'utilisateur.")
    except Exception as e:
        print(f"Erreur lors de l'initialisation: {e}")

    finally:
        # Clean up traffic
        print('\nDestroying traffic...')
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])
        time.sleep(0.5)

if __name__ == "__main__":
    main()
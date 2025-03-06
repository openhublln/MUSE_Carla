from queue import Queue, Empty
import os
import carla
import random
import numpy as np
import time
import yaml
import math
import json
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

        if isinstance(sensor_data, carla.IMUMeasurement):
            imu_data = {
                "timestamp": timestamp,
                "accelerometer": {
                    "x": sensor_data.accelerometer.x,
                    "y": sensor_data.accelerometer.y,
                    "z": sensor_data.accelerometer.z
                },
                "gyroscope": {
                    "x": sensor_data.gyroscope.x,
                    "y": sensor_data.gyroscope.y,
                    "z": sensor_data.gyroscope.z
                },
                "compass": sensor_data.compass
            }
            with open(os.path.join(sensor_folder, f"{timestamp}.json"), 'w') as f:
                json.dump(imu_data, f, indent=2)

        elif isinstance(sensor_data, carla.GnssMeasurement):
            # Save GNSS data as JSON
            gnss_data = {
                "timestamp": timestamp,
                "latitude": sensor_data.latitude,
                "longitude": sensor_data.longitude,
                "altitude": sensor_data.altitude
            }
            with open(os.path.join(sensor_folder, f"{timestamp}.json"), 'w') as f:
                json.dump(gnss_data, f, indent=2)

        elif isinstance(sensor_data, carla.Image):
            img_filename = f"{timestamp}.png"
            img_path = os.path.join(sensor_folder, img_filename)
            
            # Get the sensor's blueprint ID from the config
            blueprint_id = None
            with open('config.yml', 'r') as f:
                config = yaml.safe_load(f)
                for sensor in config["sensors"]:
                    if sensor["name"] == sensor_name:
                        blueprint_id = sensor["blueprint"]
                        break
            
            # Use blueprint ID to determine the type of camera
            if blueprint_id == "sensor.camera.semantic_segmentation":
                sensor_data.save_to_disk(img_path, carla.ColorConverter.CityScapesPalette)
            elif blueprint_id == "sensor.camera.instance_segmentation":
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
        
        # Get all files with supported extensions
        files = []
        for f in os.listdir(sensor_folder):
            # Skip instance segmentation files as they're paired with RGB
            if "instance" in sensor_folder:
                continue
                
            # Accept all supported file types
            if any(f.endswith(ext) for ext in ['.png', '.npy', '.json', '.ply']):
                # For PLY files, also check if corresponding NPY exists
                if f.endswith('.ply'):
                    npy_file = f[:-4] + '.npy'
                    if not os.path.exists(os.path.join(sensor_folder, npy_file)):
                        continue
                files.append(f)
        
        # Extract timestamps (without extension)
        ts_set = set(os.path.splitext(f)[0] for f in files)
        if ts_set:  # Only add if we found files
            ts_dict[sensor] = ts_set

    if not ts_dict:
        print(f"Warning: No valid files found in {scene_path}")
        return

    # Find timestamps common to all sensors
    common_ts = set.intersection(*ts_dict.values())
    
    if not common_ts:
        print(f"Warning: No common timestamps found in {scene_path}")
        return

    print(f"Found {len(common_ts)} common timestamps across all sensors")

    # Only delete files that aren't in common timestamps
    deleted_count = 0
    for sensor in sensor_names:
        sensor_folder = os.path.join(scene_path, sensor)
        if not os.path.isdir(sensor_folder):
            continue
            
        for file_name in os.listdir(sensor_folder):
            base_name = os.path.splitext(file_name)[0]
            if base_name not in common_ts:
                file_path = os.path.join(sensor_folder, file_name)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} non-synchronized files")
    else:
        print("All files are properly synchronized")

def setup_traffic(client, world, traffic_config):
    """Configure and spawn traffic (vehicles and pedestrians)"""
    try:
        # Get blueprints for vehicles and pedestrians
        blueprints = world.get_blueprint_library().filter('vehicle.*')
        if traffic_config.get("safe_spawn", True):
            blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']
        blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')

        # Get spawn points for vehicles
        spawn_points = world.get_map().get_spawn_points()
        num_spawn_points = len(spawn_points)

        # Get numbers from config (remove default values)
        num_vehicles = traffic_config["num_vehicles"] 
        num_pedestrians = traffic_config["num_pedestrians"]  
        
        # Limit vehicles to available spawn points
        if num_vehicles > num_spawn_points:
            print(f"Warning: Requested {num_vehicles} vehicles but only {num_spawn_points} spawn points available")
            num_vehicles = num_spawn_points

        # -------------
        # Spawn Vehicles
        # -------------
        batch = []
        vehicle_list = []
        random.shuffle(spawn_points)
        for n, transform in enumerate(spawn_points):
            if n >= num_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(carla.command.SpawnActor(blueprint, transform)
                .then(carla.command.SetAutopilot(carla.command.FutureActor, True)))

        # Apply batch spawn for vehicles
        for response in client.apply_batch_sync(batch, True):
            if response.error:
                print(f"Error spawning vehicle: {response.error}")
            else:
                vehicle_list.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # Some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        
        # 1. Take all the random locations to spawn
        spawn_points = []
        for i in range(num_pedestrians):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_point.location.z += 1  # Spawn slightly above ground
                spawn_points.append(spawn_point)

        # 2. Spawn the walker objects
        batch = []
        walker_speed = []
        walkers_list = []
        
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # Set the max speed
            if walker_bp.has_attribute('speed'):
                if random.random() > percentagePedestriansRunning:
                    # Walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # Running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                walker_speed.append(0.0)
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

        # 2.1 Apply batch spawn
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if not results[i].error:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2

        # 3. Spawn the walker controllers
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))

        # 3.1 Apply batch spawn
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if not results[i].error:
                walkers_list[i]["con"] = results[i].actor_id

        # 4. Initialize each controller and set target to walk to
        walker_controller_list = []
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(len(walkers_list)):
            controller_id = walkers_list[i]["con"]
            controller = world.get_actor(controller_id)
            if controller is None:
                continue
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(float(walker_speed[i]))
            walker_controller_list.append(controller_id)

        # Set up traffic manager for vehicles
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.global_percentage_speed_difference(30.0)

        # Set vehicle lights if specified
        if traffic_config.get("car_lights_on", False):
            for vehicle_id in vehicle_list:
                vehicle = world.get_actor(vehicle_id)
                if vehicle is not None:
                    vehicle.set_light_state(carla.VehicleLightState.All)

        all_ids = vehicle_list + [w["id"] for w in walkers_list] + walker_controller_list
        print(f"Successfully spawned {len(vehicle_list)} vehicles and {len(walkers_list)} walkers")
        return all_ids

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

def spawn_ego_vehicle(world, blueprint_library, traffic_manager, max_retries=10):
    """Safely spawn the ego vehicle by trying different spawn points"""
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)  # Randomize spawn points
    
    bp = blueprint_library.find('vehicle.lincoln.mkz')
    bp.set_attribute('role_name', 'ego')
    
    for retry in range(max_retries):
        if retry > 0:
            print(f"Retrying ego vehicle spawn (attempt {retry + 1}/{max_retries})")
        
        # Try each spawn point until we find one that works
        for spawn_point in spawn_points:
            try:
                vehicle = world.try_spawn_actor(bp, spawn_point)
                if vehicle is not None:
                    # Successfully spawned - set autopilot and return
                    vehicle.set_autopilot(True, traffic_manager.get_port())
                    print(f"Successfully spawned ego vehicle after {retry + 1} attempts")
                    return vehicle
            except Exception as e:
                continue
        
        # If we get here, no spawn points worked - wait a bit and try again
        print("All spawn points blocked, waiting for clearance...")
        world.tick()  # Tick the world to update physics
        time.sleep(0.5)
    
    raise RuntimeError(f"Failed to spawn ego vehicle after {max_retries} attempts")

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
        traffic_config = sim_config.get("traffic", {}) 

        num_scenes = sim_config["num_scenes"]
        seconds_per_scene = sim_config["seconds_per_scene"]
        ticks_per_scene = seconds_per_scene * 20  # 20Hz simulation
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
            scene_completed = False
            max_scene_retries = 3  # Maximum number of retries per scene
            
            for scene_retry in range(max_scene_retries):
                try:
                    # Create folders and sensor queue
                    sensor_names = [s["name"] for s in sensors_config]
                    save_path = create_scene_folders(scene_id, sensor_names, base_save_path)
                    scene_paths.append(save_path)
                    sensor_queue = Queue()
                    blueprint_library = world.get_blueprint_library()

                    # Try to spawn ego vehicle with retries
                    print(f"\nScene {scene_id} - Attempt {scene_retry + 1}/{max_scene_retries}")
                    try:
                        vehicle = spawn_ego_vehicle(world, blueprint_library, traffic_manager)
                    except RuntimeError as e:
                        print(f"Failed to spawn ego vehicle: {e}")
                        if scene_retry < max_scene_retries - 1:
                            print("Retrying scene...")
                            continue
                        else:
                            print("Max retries reached, skipping scene")
                            break

                    print("Ego vehicle spawned, waiting for stabilization...")
                    for _ in range(20):
                        world.tick()

                    # Setup sensors and run simulation
                    sensor_list = []
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

                    # Run simulation
                    run_simulation(scene_id, world, vehicle, sensor_list, sensor_queue, ticks_per_scene)
                    scene_completed = True
                    break  # Scene completed successfully, exit retry loop

                except Exception as e:
                    print(f"Error during scene {scene_id}: {e}")
                    if scene_retry < max_scene_retries - 1:
                        print("Retrying scene...")
                        continue
                    else:
                        print("Max retries reached, skipping scene")
                finally:
                    # Cleanup if needed
                    if 'vehicle' in locals() and vehicle is not None and vehicle.is_alive:
                        vehicle.destroy()
                    if 'sensor_list' in locals():
                        for sensor in sensor_list:
                            if sensor.is_alive:
                                sensor.destroy()

            if not scene_completed:
                print(f"Failed to complete scene {scene_id} after {max_scene_retries} attempts")

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
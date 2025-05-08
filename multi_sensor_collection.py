import os
import random
import time
import yaml
import carla
import json
from queue import Queue, Empty
from bounding_box_export import export_3d_bboxes
from traffic_setup import setup_traffic, spawn_ego_vehicle
from sensor_processing import process_sensor_config, sensor_callback, clean_scene_data
from simulation_logic import run_simulation, create_scene_folders
from generate_bbox_annotations import process_scene

EGO_POSE_FOLDER = "ego_pose"
LOG_INFO_FILENAME = "log_info.json"

def save_ego_pose(ego_vehicle, timestamp, ego_pose_dir):
    """Save the ego vehicle's pose as a JSON file in the ego_pose directory."""
    transform = ego_vehicle.get_transform()
    pose = {
        "timestamp": timestamp,
        "translation": {
            "x": transform.location.x,
            "y": transform.location.y,
            "z": transform.location.z
        },
        "rotation": {
            "pitch": transform.rotation.pitch,
            "yaw": transform.rotation.yaw,
            "roll": transform.rotation.roll
        }
    }
    os.makedirs(ego_pose_dir, exist_ok=True)
    pose_path = os.path.join(ego_pose_dir, f"{timestamp}.json")
    with open(pose_path, 'w') as f:
        json.dump(pose, f, indent=2)

def collect_log_info(world, ego_vehicle, base_save_path):
    """Collects and saves simulation log info for NuScenes log.json generation."""
    map_name = world.get_map().name.split('/')[-1]
    vehicle_blueprint = ego_vehicle.type_id
    date_captured = time.strftime("%Y-%m-%d")
    logfile = f"carla_log_{date_captured}.log"
    log_info = {
        "logfile": logfile,
        "vehicle": vehicle_blueprint,
        "date_captured": date_captured,
        "location": map_name
    }
    log_info_path = os.path.join(base_save_path, LOG_INFO_FILENAME)
    with open(log_info_path, 'w') as f:
        json.dump(log_info, f, indent=2)

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
        settings.fixed_delta_seconds = 0.05  # 20Hz simulation
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

        log_info_collected = False
        ego_vehicle_for_log = None

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

                    # Collect log info if not already collected
                    if not log_info_collected:
                        collect_log_info(world, vehicle, base_save_path)
                        log_info_collected = True
                        ego_vehicle_for_log = vehicle

                    # Setup ego_pose directory
                    ego_pose_dir = os.path.join(save_path, EGO_POSE_FOLDER)

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
                        actor.listen(lambda data, q=sensor_queue, name=sensor["name"], 
                                    path=save_path, w=world, v=vehicle, s=actor:
                                    sensor_callback(data, q, name, path, w, v, s))

                    # Run simulation and collect ego pose at each tick
                    print(f"\nStarting data collection for scene {scene_id}...")
                    print(f"Expected frames: {ticks_per_scene} (20Hz for {seconds_per_scene} seconds)")
                    
                    for tick in range(ticks_per_scene):
                        world.tick()
                        snapshot = world.get_snapshot()
                        timestamp = int(snapshot.timestamp.elapsed_seconds * 1000)
                        save_ego_pose(vehicle, timestamp, ego_pose_dir)
                        
                        # Add a small delay between ticks to ensure sensor callbacks complete
                        time.sleep(0.01)  # 10ms delay between ticks
                        
                        if tick % 10 == 0:  # Progress update every 10 frames
                            print(f"Collected {tick + 1}/{ticks_per_scene} frames")

                    # Wait for all sensor callbacks to complete
                    print("Waiting for sensor callbacks to complete...")
                    expected_calls = len(sensor_list) * ticks_per_scene
                    completed_calls = 0
                    timeout_counter = 0
                    max_timeouts = 10  # Increased max timeouts
                    
                    while completed_calls < expected_calls and timeout_counter < max_timeouts:
                        try:
                            timestamp, sensor_name = sensor_queue.get(timeout=5.0)  # Increased timeout
                            if timestamp > 0:  # Valid timestamp
                                completed_calls += 1
                                if completed_calls % 100 == 0:  # Progress update every 100 calls
                                    print(f"Completed {completed_calls}/{expected_calls} sensor callbacks")
                        except Empty:
                            timeout_counter += 1
                            print(f"Warning: Timeout {timeout_counter}/{max_timeouts} waiting for sensor callback. Completed {completed_calls}/{expected_calls}")
                            if timeout_counter >= max_timeouts:
                                print("Max timeouts reached, proceeding with cleanup")
                                break

                    # Add a longer delay before cleanup to ensure all file operations are complete
                    print("Waiting for file operations to complete...")
                    time.sleep(5.0)  # Increased wait time
                    
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
                    # Add a longer delay before cleanup to allow pending operations
                    print("Waiting before cleanup...")
                    time.sleep(5.0)  # Increased wait time
                    
                    # Cleanup if needed
                    if 'vehicle' in locals() and vehicle is not None:
                        try:
                            vehicle.get_transform()  # Check if vehicle is still alive
                            vehicle.destroy()
                        except Exception:
                            print("Vehicle already destroyed")
                            
                    if 'sensor_list' in locals():
                        for sensor in sensor_list:
                            try:
                                sensor.get_transform()  # Check if sensor is still alive
                                sensor.destroy()
                            except Exception:
                                print(f"Sensor {sensor.type_id} already destroyed")
                    
                    # Add another delay after cleanup
                    print("Waiting after cleanup...")
                    time.sleep(5.0)  # Increased wait time

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
import os
import json
import math
import yaml
import numpy as np
import carla
from bounding_box_export import export_3d_bboxes

def calculate_radar_intensity(depth):
    """Calcule l'intensité du signal radar."""
    rcs = 10          # Section efficace radar moyenne (m²)
    noise_floor = 1e-9
    ref_distance = 10 # Distance de référence (m)
    intensity = (ref_distance / depth) ** 4 * rcs
    noise = np.random.normal(0, noise_floor)
    return max(intensity + noise, 0)

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

def sensor_callback(sensor_data, sensor_queue, sensor_name, save_path, world=None, ego_vehicle=None, sensor_actor=None):
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
            
            # New: Call export to write 3D bounding boxes (if sensor_actor provided)
            if sensor_actor is not None:
                export_3d_bboxes(sensor_data, sensor_folder, world, ego_vehicle, sensor_actor)

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
                # Skip 3D bbox annotation files when building timestamp list
                if "_3dbbox.json" in f:
                    continue
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
            # Preserve 3D bbox annotation files if their corresponding frame exists
            if "_3dbbox.json" in file_name:
                base_ts = file_name.split("_3dbbox.json")[0]
                if base_ts in common_ts:
                    continue
                
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
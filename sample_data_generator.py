import json
from pathlib import Path
from nuscene_utils import generate_token, euler_to_quaternion, adjust_z_for_ego_pose
from PIL import Image
import shutil
import numpy as np

class SampleDataGenerator:
    def __init__(self, converter):
        self.converter = converter

    def generate_sample_data_entries(self, scene_folder: str, scene_token: str):
        scene_path = self.converter.input_base / scene_folder
        sensor_json_path = self.converter.output_base / self.converter.version / 'sensor.json'
        calibrated_json_path = self.converter.output_base / self.converter.version / 'calibrated_sensor.json'
        
        try:
            with open(sensor_json_path, 'r') as f:
                sensor_json = json.load(f)
            with open(calibrated_json_path, 'r') as f:
                calibrated_json = json.load(f)
        except FileNotFoundError as e:
            print(f"Error: Required JSON file not found for sample_data generation: {e}")
            return

        channel_to_sensor_token = {entry["channel"]: entry["token"] for entry in sensor_json}
        sensor_token_to_calibrated_token = {entry["sensor_token"]: entry["token"] for entry in calibrated_json}
        
        sensor_folder_to_channel = {}
        sensor_mappings = self.converter.config.get("sensor_mappings", {})
        for sensor_type, mapping in sensor_mappings.items():
            for folder, channel in mapping.items():
                sensor_folder_to_channel[folder] = channel
        
        simulation_sensors = self.converter.sim_config.get("sensors", [])
        
        scene_samples = [entry for entry in self.converter.samples if entry["scene_token"] == scene_token]
        sample_timestamps = {entry["timestamp"]: entry["token"] for entry in scene_samples}
        keyframe_timestamps = set(sample_timestamps.keys())
        
        all_sensor_timestamps = set()
        for sensor in simulation_sensors:
            sensor_name = sensor["name"]
            sensor_data_folder = scene_path / sensor_name
            if not sensor_data_folder.exists() or not sensor_data_folder.is_dir():
                continue

            ext = ""
            if sensor["type"] == "camera":
                ext = ".png"
            elif sensor["type"] in ("lidar", "semantic_lidar", "radar"):
                ext = ".npy"
            elif sensor["type"] in ("imu", "gnss"):
                ext = ".json"
            else:
                continue
            files = [f for f in sensor_data_folder.glob(f"*{ext}") if f.is_file()]
            for file in files:
                try:
                    timestamp = int(file.stem)
                    all_sensor_timestamps.add(timestamp)
                except ValueError:
                    # Attempt to parse if filename is <timestamp>_suffix.ext
                    try:
                        timestamp = int(file.stem.split('_')[0])
                        all_sensor_timestamps.add(timestamp)
                    except (ValueError, IndexError):
                        continue

        ego_pose_dir = scene_path / self.converter.EGO_POSE_FOLDER
        ego_pose_timestamps = {}
        if ego_pose_dir.exists():
            for pose_file in ego_pose_dir.glob("*.json"):
                try:
                    timestamp = int(pose_file.stem)
                    with open(pose_file, 'r') as f:
                        pose_data = json.load(f)
                        
                    translation = [
                        float(pose_data["translation"]["x"]),
                        float(pose_data["translation"]["y"]),
                        float(pose_data["translation"]["z"])
                    ]
                    
                    # Convert CARLA Euler angles to NuScenes quaternion
                    rotation = euler_to_quaternion(
                        float(pose_data["rotation"]["roll"]),
                        float(pose_data["rotation"]["pitch"]),
                        float(pose_data["rotation"]["yaw"])
                    )
                    
                    # Adjust Z coordinate to match NuScenes' Z=0 assumption
                    translation = adjust_z_for_ego_pose(translation)
                    
                    token = generate_token()
                    ego_pose_entry = {
                        "token": token,
                        "timestamp": timestamp,
                        "translation": translation,
                        "rotation": rotation
                    }
                    
                    self.converter.ego_poses.append(ego_pose_entry)
                    ego_pose_timestamps[timestamp] = token
                    
                except Exception as e:
                    print(f"Warning: Error processing ego pose file {pose_file}: {e}")
                    continue

        for sensor in simulation_sensors:
            sensor_name = sensor["name"]
            sensor_type = sensor["type"]
            channel = sensor_folder_to_channel.get(sensor_name)
            if not channel:
                continue
            
            sensor_token_val = channel_to_sensor_token.get(channel, "")
            calibrated_sensor_token = sensor_token_to_calibrated_token.get(sensor_token_val, "")
            current_sensor_folder = scene_path / sensor_name

            if not current_sensor_folder.exists() or not current_sensor_folder.is_dir():
                continue

            fileformat, width, height, ext = "", None, None, ""
            if sensor_type == "camera":
                ext = ".png"
                fileformat = "jpg"  # Changed from png to jpg for nuScenes compatibility
                width = int(float(sensor.get("attributes", {}).get("image_size_x", 0)))
                height = int(float(sensor.get("attributes", {}).get("image_size_y", 0)))
            elif sensor_type == "lidar":
                ext = ".npy"
                fileformat = "bin"  # Use .bin for LIDAR in nuScenes
            elif sensor_type == "radar":
                ext = ".npy"
                fileformat = "pcd"  # Use .pcd for RADAR in nuScenes
            elif sensor_type == "semantic_lidar":
                ext = ".npy"
                fileformat = "bin"  # Use .bin for semantic LIDAR
            elif sensor_type in ("imu", "gnss"):
                ext = ".json"
                fileformat = "json"
            else:
                continue
            
            files = sorted([f for f in current_sensor_folder.glob(f"*{ext}") if f.is_file()])
            if not files:
                continue
            
            # Create target directory for this sensor
            target_dir = self.converter.output_base / f"samples/{channel}"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            entries = []
            for file in files:
                try:
                    timestamp = 0
                    try:
                        timestamp = int(file.stem)
                    except ValueError:
                        try:
                            timestamp = int(file.stem.split('_')[0])
                        except (ValueError, IndexError):
                            continue
                    
                    closest_keyframe_ts = min(keyframe_timestamps, key=lambda x: abs(x - timestamp), default=None)
                    if closest_keyframe_ts is None:
                        continue
                    sample_token = sample_timestamps[closest_keyframe_ts]
                    
                    is_key_frame = timestamp in keyframe_timestamps
                    ego_pose_token = ego_pose_timestamps.get(timestamp, "")
                    
                    # Create the correct filename for nuScenes format
                    if sensor_type == "camera":
                        filename = f"samples/{channel}/{timestamp}.jpg"
                    elif sensor_type == "lidar":
                        filename = f"samples/{channel}/{timestamp}.bin"
                    elif sensor_type == "radar":
                        filename = f"samples/{channel}/{timestamp}.pcd"
                    elif sensor_type == "semantic_lidar":
                        filename = f"samples/{channel}/{timestamp}.bin"
                    else:
                        filename = f"samples/{channel}/{timestamp}{ext}"
                    
                    # Copy the file to the target directory
                    target_file = target_dir / filename.split('/')[-1]  # Use just the filename for the target path
                    if sensor_type == "camera":
                        # Convert PNG to JPG for camera images
                        img = Image.open(file)
                        img.convert('RGB').save(target_file, 'JPEG')
                        print(f"Converted and copied {file.name} to {target_file.name}")
                    elif sensor_type == "lidar":
                        # Convert NPY to BIN for LIDAR
                        points = np.load(file)
                        points.astype(np.float32).tofile(target_file)
                        print(f"Converted {file.name} to {target_file.name}")
                    elif sensor_type == "radar":
                        # Convert NPY to PCD for RADAR
                        points = np.load(file)
                        # Ensure 18-column format for RADAR
                        if points.shape[1] < 18:
                            padded_points = np.zeros((points.shape[0], 18))
                            padded_points[:, :points.shape[1]] = points
                            points = padded_points
                        
                        # Write binary PCD file
                        with open(target_file, 'wb') as f:
                            f.write(b"# .PCD v0.7 - Point Cloud Data file format\n")
                            f.write(b"VERSION 0.7\n")
                            f.write(b"FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms\n")
                            f.write(b"SIZES 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1\n")
                            f.write(b"TYPES F F F I I F F F F F I I I I I I I I\n")
                            f.write(b"COUNTS 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n")
                            f.write(f"WIDTH {points.shape[0]}\n".encode())
                            f.write(b"HEIGHT 1\n")
                            f.write(b"VIEWPOINT 0 0 0 1 0 0 0\n")
                            f.write(f"POINTS {points.shape[0]}\n".encode())
                            f.write(b"DATA binary\n")
                            points.astype(np.float32).tofile(f)
                        print(f"Converted {file.name} to {target_file.name}")
                    elif sensor_type == "semantic_lidar":
                        # Convert NPY to BIN for semantic LIDAR
                        points = np.load(file)
                        points.astype(np.float32).tofile(target_file)
                        print(f"Converted {file.name} to {target_file.name}")
                    else:
                        shutil.copy2(file, target_file)
                        print(f"Copied {file.name} to {target_file.name}")
                    
                    token = generate_token()
                    entry = {
                        "token": token,
                        "sample_token": sample_token,
                        "ego_pose_token": ego_pose_token,
                        "calibrated_sensor_token": calibrated_sensor_token,
                        "filename": filename,
                        "fileformat": fileformat,
                        "width": width,
                        "height": height,
                        "timestamp": timestamp,
                        "is_key_frame": is_key_frame,
                        "next": "",
                        "prev": ""
                    }
                    entries.append(entry)
                except Exception as e:
                    print(f"Warning: Error processing file {file} for sensor {sensor_name}: {e}")
                    continue
            
            for i, entry in enumerate(entries):
                if i > 0:
                    entry["prev"] = entries[i-1]["token"]
                if i < len(entries) - 1:
                    entry["next"] = entries[i+1]["token"]
                self.converter.sample_data.append(entry) 
import json
from pathlib import Path
from nuscene_utils import generate_token, euler_to_quaternion # euler_to_quaternion might be needed for ego_pose processing if not already handled

class SampleDataGenerator:
    def __init__(self, converter):
        self.converter = converter

    def generate_sample_data_entries(self, scene_folder: str, scene_token: str):
        scene_path = self.converter.input_base / scene_folder
        sensor_json_path = self.converter.output_base / 'sensor.json'
        calibrated_json_path = self.converter.output_base / 'calibrated_sensor.json'
        
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
            sensor_data_folder = scene_path / sensor_name # Corrected variable name
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
            files = [f for f in sensor_data_folder.glob(f"*{ext}") if f.is_file()] # Corrected variable name
            for file in files:
                try:
                    timestamp = int(file.stem) # Assumes filename is just the timestamp for sensor data
                    all_sensor_timestamps.add(timestamp)
                except ValueError:
                    # Attempt to parse if filename is <timestamp>_suffix.ext
                    try:
                        timestamp = int(file.stem.split('_')[0])
                        all_sensor_timestamps.add(timestamp)
                    except (ValueError, IndexError):
                        # print(f"Warning: Could not parse timestamp from sensor data file {file}")
                        continue 

        ego_pose_dir = scene_path / self.converter.EGO_POSE_FOLDER
        ego_pose_timestamps = {}
        if ego_pose_dir.exists() and ego_pose_dir.is_dir():
            for pose_file in ego_pose_dir.glob("*.json"):
                try:
                    with open(pose_file, "r") as f:
                        pose_data = json.load(f)
                    timestamp = pose_data["timestamp"]
                    
                    translation = [
                        float(pose_data["translation"]["x"]),
                        float(pose_data["translation"]["y"]),
                        0.0  # Force Z=0 for ego pose
                    ]
                    roll = float(pose_data["rotation"]["roll"])
                    pitch = float(pose_data["rotation"]["pitch"])
                    yaw = float(pose_data["rotation"]["yaw"])
                    quaternion = euler_to_quaternion(roll, pitch, yaw) # Using utility
                    token = generate_token()
                    ego_pose_entry = {
                        "token": token,
                        "translation": translation,
                        "rotation": quaternion,
                        "timestamp": timestamp
                    }
                    self.converter.ego_poses.append(ego_pose_entry)
                    ego_pose_timestamps[timestamp] = token
                except Exception as e:
                    # print(f"Warning: Error processing ego pose file {pose_file}: {e}")
                    continue
        
        for sensor in simulation_sensors:
            sensor_name = sensor["name"]
            sensor_type = sensor["type"]
            channel = sensor_folder_to_channel.get(sensor_name)
            if not channel:
                continue
            
            sensor_token_val = channel_to_sensor_token.get(channel, "") # Renamed to avoid conflict
            calibrated_sensor_token = sensor_token_to_calibrated_token.get(sensor_token_val, "")
            current_sensor_folder = scene_path / sensor_name # Corrected variable name

            if not current_sensor_folder.exists() or not current_sensor_folder.is_dir():
                continue

            fileformat, width, height, ext = "", None, None, ""
            if sensor_type == "camera":
                ext = ".png"
                fileformat = "png"
                width = int(float(sensor.get("attributes", {}).get("image_size_x", 0)))
                height = int(float(sensor.get("attributes", {}).get("image_size_y", 0)))
            elif sensor_type in ("lidar", "semantic_lidar", "radar"):
                ext = ".npy"
                fileformat = "npy"
            elif sensor_type in ("imu", "gnss"):
                ext = ".json"
                fileformat = "json"
            else:
                continue
            
            files = sorted([f for f in current_sensor_folder.glob(f"*{ext}") if f.is_file()])
            if not files:
                continue
            
            entries = []
            for file in files:
                try:
                    timestamp = 0
                    try:
                        timestamp = int(file.stem) # Assumes filename is just the timestamp
                    except ValueError:
                         # Attempt to parse if filename is <timestamp>_suffix.ext
                        try:
                            timestamp = int(file.stem.split('_')[0])
                        except (ValueError, IndexError):
                            # print(f"Warning: Could not parse timestamp from data file {file} for sensor {sensor_name}")
                            continue
                    
                    closest_keyframe_ts = min(keyframe_timestamps, key=lambda x: abs(x - timestamp), default=None)
                    if closest_keyframe_ts is None:
                        # print(f"Warning: No keyframes found for timestamp {timestamp} of file {file}")
                        continue
                    sample_token = sample_timestamps[closest_keyframe_ts]
                    
                    is_key_frame = timestamp in keyframe_timestamps
                    ego_pose_token = ego_pose_timestamps.get(timestamp, "")
                    
                    token = generate_token()
                    entry = {
                        "token": token,
                        "sample_token": sample_token,
                        "ego_pose_token": ego_pose_token,
                        "calibrated_sensor_token": calibrated_sensor_token,
                        "filename": str(file.relative_to(self.converter.input_base)),
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
                    # print(f"Warning: Error processing file {file} for sensor {sensor_name}: {e}")
                    continue
            
            for i, entry in enumerate(entries):
                if i > 0:
                    entry["prev"] = entries[i-1]["token"]
                if i < len(entries) - 1:
                    entry["next"] = entries[i+1]["token"]
                self.converter.sample_data.append(entry) 
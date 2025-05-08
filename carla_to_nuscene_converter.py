#!/usr/bin/env python

import os
import sys
import json
import yaml
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.spatial.transform import Rotation
import carla
import math

class NuScenesConverter:
    """Converts CARLA sensor data to NuScenes format."""
    
    def _detect_scene_folders(self):
        """Automatically detect scene folders based on the simulation configuration."""
        scene_base_path = self.input_base
        detected_scenes = [folder.name for folder in scene_base_path.iterdir() if folder.is_dir()]
        return detected_scenes

    def __init__(self, config_path: str):
        """Initialize the converter with the given config file.
        
        Args:
            config_path: Path to the converter_config.yml file
        """
        self.config = self._load_config(config_path)
        self.version = self.config['version']
        self.input_base = Path(self.config['input']['base_dir'])
        self.output_base = Path(self.config['output']['base_dir'])
        self.scene_folders = self._detect_scene_folders()  # Automatically detect scenes

        # Ensure output directory exists
        self.output_base.mkdir(parents=True, exist_ok=True)

        # Initialize storage for various records
        self._init_storage()

        # Parse configuration files
        self._parse_config_files()

        # Set global reference frame
        self._set_global_reference_frame()
        
        # Initialize attribute name to token mapping
        self.attribute_name_to_token = {}

    def _load_config(self, config_path: str) -> dict:
        """Load and validate the converter configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # TODO: Add config validation
        return config
    
    def _init_storage(self):
        """Initialize empty storage for all NuScenes tables."""
        self.attributes = []
        self.calibrated_sensors = []
        self.categories = []
        self.ego_poses = []
        self.instances = []
        self.logs = []
        self.maps = []
        self.samples = []
        self.sample_annotations = []
        self.sample_data = []
        self.scenes = []
        self.sensors = []
        self.visibilities = []
        
        # Token mappings for linking records
        self.token_maps = {
            'calibrated_sensor': {},  # sensor_name -> token
            'instance': {},  # (scene_token, actor_id) -> token
            'sample': {},  # timestamp -> token
            'scene': {},  # scene_name -> token
        }
    
    def _generate_token(self) -> str:
        """Generate a unique token using UUID."""
        return uuid.uuid4().hex

    def _generate_composite_token(self, *args: Any) -> str:
        """Generate a composite token based on input arguments.

        Args:
            *args: Components to include in the composite token.

        Returns:
            A unique token string.
        """
        composite_string = "_".join(map(str, args))
        return uuid.uuid5(uuid.NAMESPACE_DNS, composite_string).hex

    def _link_tokens(self, scene_id: str, actor_id: Optional[int] = None, sensor_name: Optional[str] = None) -> str:
        """Maintain mappings between CARLA IDs and NuScenes tokens.

        Args:
            scene_id: The ID of the scene.
            actor_id: The ID of the actor (optional).
            sensor_name: The name of the sensor (optional).

        Returns:
            The generated or existing token for the given identifiers.
        """
        key = (scene_id, actor_id, sensor_name)
        if key not in self.token_maps['instance']:
            self.token_maps['instance'][key] = self._generate_composite_token(scene_id, actor_id, sensor_name)
        return self.token_maps['instance'][key]

    def _populate_foreign_keys(self, record: dict, scene_id: str, actor_id: Optional[int] = None, sensor_name: Optional[str] = None):
        """Ensure foreign key relationships are correctly populated.

        Args:
            record: The record to populate with foreign keys.
            scene_id: The ID of the scene.
            actor_id: The ID of the actor (optional).
            sensor_name: The name of the sensor (optional).
        """
        record['scene_token'] = self._link_tokens(scene_id)
        if actor_id is not None:
            record['instance_token'] = self._link_tokens(scene_id, actor_id)
        if sensor_name is not None:
            record['sensor_token'] = self._link_tokens(scene_id, None, sensor_name)

    def _parse_config_files(self):
        """Parse the simulation and converter configuration files."""
        # Load simulation config
        sim_config_path = self.input_base / 'config.yml'
        if not sim_config_path.exists():
            sim_config_path = Path(__file__).parent / 'config.yml'  # Fallback to the correct location
        with open(sim_config_path, 'r') as f:
            self.sim_config = yaml.safe_load(f)

        # Load converter config (already loaded in __init__)
        self.converter_config = self.config

    def _set_global_reference_frame(self):
        """Define the global reference frame for the dataset."""
        # Use CARLA map origin as the global reference frame
        self.global_origin = np.array([0.0, 0.0, 0.0])

    def _transform_to_global_frame(self, transform):
        """Transform CARLA world coordinates to the global reference frame.

        Args:
            transform: A CARLA Transform object containing location and rotation.

        Returns:
            A dictionary with `translation` and `rotation` in the global frame.
        """
        # Extract location and rotation from the transform
        location = transform.location
        rotation = transform.rotation

        # Convert location to global frame
        translation = np.array([location.x, location.y, location.z]) - self.global_origin

        # Convert rotation (Euler angles) to Quaternion
        quaternion = Rotation.from_euler('xyz', [np.radians(rotation.roll), np.radians(rotation.pitch), np.radians(rotation.yaw)]).as_quat()

        return {
            "translation": translation.tolist(),
            "rotation": quaternion.tolist()
        }

    def _adjust_z_for_ego_pose(self, translation):
        """Adjust the Z-coordinate for ego_pose to match NuScenes' Z=0 assumption.

        Args:
            translation: A list representing the [x, y, z] translation.

        Returns:
            A modified translation with Z set to 0.
        """
        translation[2] = 0.0  # Force Z=0
        return translation

    def _find_data_files(self, scene_folder: str) -> Dict[str, List[str]]:
        """Recursively find data files in the specified scene folder.

        Args:
            scene_folder: Path to the scene folder.

        Returns:
            A dictionary where keys are sensor names and values are lists of file paths.
        """
        scene_path = self.input_base / scene_folder
        data_files = {}

        for sensor_folder in scene_path.iterdir():
            if sensor_folder.is_dir():
                sensor_name = sensor_folder.name
                files = list(sensor_folder.glob('*.*'))  # Match all files
                data_files[sensor_name] = [str(f) for f in files]

        return data_files

    def _organize_data_by_timestamp(self, data_files: Dict[str, List[str]]) -> Dict[int, Dict[str, str]]:
        """Organize data files by timestamp and sensor.

        Args:
            data_files: A dictionary of sensor names and their file paths.

        Returns:
            A dictionary where keys are timestamps and values are dictionaries
            mapping sensor names to file paths.
        """
        organized_data = {}
        total_files = 0

        for sensor_name, files in data_files.items():
            for file_path in files:
                # Extract timestamp from the file name (assumes format: <timestamp>_suffix.ext)
                try:
                    timestamp = int(Path(file_path).stem.split('_')[0])
                    total_files += 1

                    if timestamp not in organized_data:
                        organized_data[timestamp] = {}

                    organized_data[timestamp][sensor_name] = file_path
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not extract timestamp from {file_path}: {e}")
                    continue

        print(f"\nOrganized {total_files} files into {len(organized_data)} unique timestamps")
        if organized_data:
            timestamps = sorted(organized_data.keys())
            print(f"Timestamp range: {timestamps[0]} to {timestamps[-1]}")
            print(f"Average interval: {(timestamps[-1] - timestamps[0]) / (len(timestamps) - 1):.2f}ms")

        return organized_data

    def _select_keyframes(self, timestamps: List[int], target_rate: float = 2.0) -> List[int]:
        """Select keyframes at the target rate from the higher-frequency data.

        Args:
            timestamps: A sorted list of all unique timestamps.
            target_rate: The target keyframe rate in Hz.

        Returns:
            A list of selected keyframe timestamps.
        """
        if not timestamps:
            return []

        # Ensure timestamps are sorted
        timestamps.sort()

        # Calculate the interval between frames (assuming timestamps are in milliseconds)
        interval = int(1000 / target_rate)  # Convert to milliseconds
        keyframes = [timestamps[0]]  # Always include the first timestamp
        
        # Find all bbox files to ensure we prioritize frames with annotations
        bbox_timestamps = set()
        scene_path = self.input_base
        for sensor_folder in scene_path.iterdir():
            if not sensor_folder.is_dir():
                continue
            bbox_files = list(sensor_folder.glob("*_3dbbox.json"))
            for bbox_file in bbox_files:
                try:
                    if bbox_file.stat().st_size > 2:  # Skip empty files
                        ts = int(bbox_file.stem.split('_')[0])
                        bbox_timestamps.add(ts)
                except (ValueError, IndexError):
                    continue

        print(f"Found {len(bbox_timestamps)} timestamps with valid bbox annotations")
        
        last_keyframe = timestamps[0]
        for ts in timestamps[1:]:
            time_diff = ts - last_keyframe
            
            # If we've passed the interval, select the next keyframe
            if time_diff >= interval:
                # Prioritize timestamps that have bbox annotations
                candidates = [t for t in timestamps if t > last_keyframe and t <= ts]
                
                # First try to find a candidate with bbox data
                bbox_candidates = [t for t in candidates if t in bbox_timestamps]
                if bbox_candidates:
                    next_keyframe = min(bbox_candidates, key=lambda x: abs(x - (last_keyframe + interval)))
                else:
                    # If no bbox data available, just take the closest to the ideal interval
                    next_keyframe = min(candidates, key=lambda x: abs(x - (last_keyframe + interval)))
                
                if next_keyframe not in keyframes:
                    keyframes.append(next_keyframe)
                    last_keyframe = next_keyframe

        print(f"Selected {len(keyframes)} keyframes from {len(timestamps)} total timestamps")
        print(f"Keyframe timestamps: {keyframes}")
        return keyframes

    def _generate_sample_entries(self, keyframes: List[int], scene_token: str) -> List[dict]:
        """Generate `sample.json` entries for the selected keyframes for a single scene.

        Args:
            keyframes: A list of selected keyframe timestamps for this scene.
            scene_token: The token of the scene to which these samples belong.

        Returns:
            A list of sample dictionaries generated for this scene.
        """
        scene_samples = []
        for i, timestamp in enumerate(keyframes):
            sample_token = self._generate_token()
            sample = {
                "token": sample_token,
                "timestamp": timestamp,
                "scene_token": scene_token,
                "prev": scene_samples[-1]["token"] if i > 0 else "",  # Link to previous sample
                "next": "",  # Will be updated in next iteration
            }
            # Update previous sample's next token
            if i > 0:
                scene_samples[-1]["next"] = sample_token
            
            scene_samples.append(sample)
            # Add token mapping for easy lookup if needed later
            self.token_maps['sample'][timestamp] = sample_token

        return scene_samples

    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> List[float]:
        """Convert Euler angles (roll, pitch, yaw) to a Quaternion (w, x, y, z).

        Args:
            roll: Rotation around the X-axis in degrees.
            pitch: Rotation around the Y-axis in degrees.
            yaw: Rotation around the Z-axis in degrees.

        Returns:
            A list representing the Quaternion [w, x, y, z].
        """
        quaternion = Rotation.from_euler('xyz', [np.radians(roll), np.radians(pitch), np.radians(yaw)]).as_quat()
        return quaternion.tolist()

    def _convert_bounding_box_size(self, extent) -> List[float]:
        """Convert CARLA bounding box extent to NuScenes size.

        Args:
            extent: A CARLA bounding box extent object with x, y, z (half-dimensions).

        Returns:
            A list representing the size [width, length, height] in NuScenes format.
        """
        # Double the extent values to get full dimensions
        width = extent.y * 2  # CARLA y -> NuScenes width
        length = extent.x * 2  # CARLA x -> NuScenes length
        height = extent.z * 2  # CARLA z -> NuScenes height

        return [width, length, height]

    def _generate_sensor_entries(self):
        sensor_mappings = self.config.get("sensor_mappings", {})
        simulation_sensors = self.sim_config.get("sensors", [])
        self.sensors = []
        # Map sensor name to token for robust linking
        self.sensor_name_to_token = {}
        for sensor in simulation_sensors:
            sensor_name = sensor["name"]
            sensor_type = sensor["type"]
            if sensor_type in sensor_mappings and sensor_name in sensor_mappings[sensor_type]:
                channel = sensor_mappings[sensor_type][sensor_name]
                token = self._generate_token()
                sensor_entry = {
                    "token": token,
                    "channel": channel,
                    "modality": sensor_type
                }
                self.sensors.append(sensor_entry)
                self.sensor_name_to_token[sensor_name] = token
        sensor_output_path = self.output_base / 'sensor.json'
        with open(sensor_output_path, 'w') as f:
            json.dump(self.sensors, f, indent=2)

    def _generate_calibrated_sensors(self):
        self.calibrated_sensors = []
        simulation_sensors = self.sim_config.get("sensors", [])
        # Build sensor name to token mapping from sensor.json
        sensor_json_path = self.output_base / 'sensor.json'
        with open(sensor_json_path, 'r') as f:
            sensor_json = json.load(f)
        sensor_name_to_token = {entry["channel"]: entry["token"] for entry in sensor_json}
        sensor_mappings = self.config.get("sensor_mappings", {})
        for sensor in simulation_sensors:
            sensor_name = sensor["name"]
            sensor_type = sensor["type"]
            if sensor_type in sensor_mappings and sensor_name in sensor_mappings[sensor_type]:
                channel = sensor_mappings[sensor_type][sensor_name]
                sensor_token = sensor_name_to_token.get(channel)
                if not sensor_token:
                    continue
                loc = sensor["transform"]["location"]
                rot = sensor["transform"]["rotation"]
                # Convert CARLA coordinates (X-forward, Y-right, Z-up) to NuScenes (X-forward, Y-left, Z-up)
                # Negate the Y-coordinate for translation
                translation = [float(loc["x"]), -float(loc["y"]), float(loc["z"])]

                # Get CARLA Euler angles (degrees)
                roll_carla = float(rot.get("roll", 0.0))
                pitch_carla = float(rot.get("pitch", 0.0))
                yaw_carla = float(rot.get("yaw", 0.0))

                # Convert CARLA Euler angles to NuScenes Euler angles (degrees)
                # Negate roll and pitch due to the flipped Y-axis
                roll_nusc = -roll_carla
                pitch_nusc = -pitch_carla
                yaw_nusc = yaw_carla

                # Convert NuScenes Euler angles to quaternion
                # The _euler_to_quaternion function expects angles in the target (NuScenes) frame convention
                rotation_quaternion = self._euler_to_quaternion(roll_nusc, pitch_nusc, yaw_nusc)

                calibrated_sensor_entry = {
                    "token": self._generate_token(),
                    "sensor_token": sensor_token,
                    "translation": translation,
                    "rotation": rotation_quaternion,
                    "camera_intrinsic": []
                }
                self.calibrated_sensors.append(calibrated_sensor_entry)
        # Write calibrated_sensor.json immediately after generating
        calibrated_output_path = self.output_base / 'calibrated_sensor.json'
        with open(calibrated_output_path, 'w') as f:
            json.dump(self.calibrated_sensors, f, indent=2)

    EGO_POSE_FOLDER = "ego_pose"

    def _generate_sample_data_entries(self, scene_folder: str, scene_token: str):
        scene_path = self.input_base / scene_folder
        sensor_json_path = self.output_base / 'sensor.json'
        calibrated_json_path = self.output_base / 'calibrated_sensor.json'
        with open(sensor_json_path, 'r') as f:
            sensor_json = json.load(f)
        with open(calibrated_json_path, 'r') as f:
            calibrated_json = json.load(f)
        # Build channel -> sensor_token
        channel_to_sensor_token = {entry["channel"]: entry["token"] for entry in sensor_json}
        # Build sensor_token -> calibrated_sensor_token
        sensor_token_to_calibrated_token = {entry["sensor_token"]: entry["token"] for entry in calibrated_json}
        # Build sensor_folder -> channel mapping from config
        sensor_folder_to_channel = {}
        sensor_mappings = self.config.get("sensor_mappings", {})
        for sensor_type, mapping in sensor_mappings.items():
            for folder, channel in mapping.items():
                sensor_folder_to_channel[folder] = channel
        simulation_sensors = self.sim_config.get("sensors", [])
        
        # Get all sample tokens for this scene and their timestamps
        scene_samples = [entry for entry in self.samples if entry["scene_token"] == scene_token]
        sample_timestamps = {entry["timestamp"]: entry["token"] for entry in scene_samples}
        keyframe_timestamps = set(sample_timestamps.keys())
        
        # Get all timestamps from sensor data files
        all_sensor_timestamps = set()
        for sensor in simulation_sensors:
            sensor_name = sensor["name"]
            sensor_folder = scene_path / sensor_name
            if not sensor_folder.exists() or not sensor_folder.is_dir():
                continue
            if sensor["type"] == "camera":
                ext = ".png"
            elif sensor["type"] in ("lidar", "semantic_lidar", "radar"):
                ext = ".npy"
            elif sensor["type"] in ("imu", "gnss"):
                ext = ".json"
            else:
                continue
            files = [f for f in sensor_folder.glob(f"*{ext}") if f.is_file()]
            for file in files:
                try:
                    timestamp = int(file.stem)
                    all_sensor_timestamps.add(timestamp)
                except Exception:
                    continue
        
        # Process ego pose entries
        ego_pose_dir = scene_path / "ego_pose"
        ego_pose_timestamps = {}
        ego_pose_files_processed = 0
        ego_pose_files_skipped = 0
        skipped_timestamps = set()
        
        if ego_pose_dir.exists() and ego_pose_dir.is_dir():
            for pose_file in ego_pose_dir.glob("*.json"):
                try:
                    with open(pose_file, "r") as f:
                        pose_data = json.load(f)
                    timestamp = pose_data["timestamp"]
                    
                    # Create ego pose entry regardless of sensor data
                    translation = [
                        float(pose_data["translation"]["x"]),
                        float(pose_data["translation"]["y"]),
                        0.0  # Force Z=0 for ego pose
                    ]
                    roll = float(pose_data["rotation"]["roll"])
                    pitch = float(pose_data["rotation"]["pitch"])
                    yaw = float(pose_data["rotation"]["yaw"])
                    quaternion = self._euler_to_quaternion(roll, pitch, yaw)
                    token = self._generate_token()
                    ego_pose_entry = {
                        "token": token,
                        "translation": translation,
                        "rotation": quaternion,
                        "timestamp": timestamp
                    }
                    self.ego_poses.append(ego_pose_entry)
                    ego_pose_timestamps[timestamp] = token
                    ego_pose_files_processed += 1
                    
                    # Log if this timestamp doesn't have sensor data
                    if timestamp not in all_sensor_timestamps:
                        skipped_timestamps.add(timestamp)
                        ego_pose_files_skipped += 1
                        
                except Exception as e:
                    print(f"Warning: Error processing ego pose file {pose_file}: {e}")
                    continue
        
        print(f"\nEgo pose processing summary for scene {scene_folder}:")
        print(f"Total ego pose files processed: {ego_pose_files_processed}")
        print(f"Ego pose files without sensor data: {ego_pose_files_skipped}")
        if skipped_timestamps:
            print(f"Skipped timestamps: {sorted(skipped_timestamps)}")
        
        print(f"\nFound {len(scene_samples)} samples for scene {scene_folder}")
        print(f"Found {len(ego_pose_timestamps)} ego pose entries")
        print(f"Found {len(all_sensor_timestamps)} sensor data timestamps")
        print(f"Keyframe timestamps: {sorted(keyframe_timestamps)}")
        
        for sensor in simulation_sensors:
            sensor_name = sensor["name"]
            sensor_type = sensor["type"]
            channel = sensor_folder_to_channel.get(sensor_name)
            if not channel:
                continue
            sensor_token = channel_to_sensor_token.get(channel, "")
            calibrated_sensor_token = sensor_token_to_calibrated_token.get(sensor_token, "")
            sensor_folder = scene_path / sensor_name
            if not sensor_folder.exists() or not sensor_folder.is_dir():
                continue
            if sensor_type == "camera":
                ext = ".png"
                fileformat = "png"
                width = int(float(sensor["attributes"].get("image_size_x", 0)))
                height = int(float(sensor["attributes"].get("image_size_y", 0)))
            elif sensor_type in ("lidar", "semantic_lidar", "radar"):
                ext = ".npy"
                fileformat = "npy"
                width = None
                height = None
            elif sensor_type in ("imu", "gnss"):
                ext = ".json"
                fileformat = "json"
                width = None
                height = None
            else:
                ext = None
                fileformat = None
                width = None
                height = None
            if not ext:
                continue
            files = sorted([f for f in sensor_folder.glob(f"*{ext}") if f.is_file()])
            if not files:
                continue
            entries = []
            for file in files:
                try:
                    timestamp = int(file.stem)
                    
                    # Find the closest keyframe sample token
                    closest_keyframe = min(keyframe_timestamps, key=lambda x: abs(x - timestamp))
                    sample_token = sample_timestamps[closest_keyframe]
                    
                    # Check if this is a keyframe
                    is_key_frame = timestamp in keyframe_timestamps
                    
                    # Get ego pose token if available
                    ego_pose_token = ego_pose_timestamps.get(timestamp, "")
                    
                    token = self._generate_token()
                    entry = {
                        "token": token,
                        "sample_token": sample_token,
                        "ego_pose_token": ego_pose_token,
                        "calibrated_sensor_token": calibrated_sensor_token,
                        "filename": str(file.relative_to(self.input_base)),
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
                    print(f"Warning: Error processing file {file}: {e}")
                    continue
                
            # Link next/prev tokens
            for i, entry in enumerate(entries):
                if i > 0:
                    entry["prev"] = entries[i-1]["token"]
                if i < len(entries) - 1:
                    entry["next"] = entries[i+1]["token"]
                self.sample_data.append(entry)

    def _generate_log_entry(self):
        """Generate the NuScenes log entry from log_info.json."""
        log_info_path = self.input_base / "log_info.json"
        if not log_info_path.exists():
            print(f"Warning: log_info.json not found at {log_info_path}. Skipping log.json generation.")
            return
        with open(log_info_path, 'r') as f:
            log_info = json.load(f)
        log_entry = {
            "token": self._generate_token(),
            "logfile": log_info.get("logfile", ""),
            "vehicle": log_info.get("vehicle", ""),
            "date_captured": log_info.get("date_captured", ""),
            "location": log_info.get("location", "")
        }
        self.logs = [log_entry]
        self.log_token = log_entry["token"]

    def _assign_log_token_to_scenes(self):
        """Assign the log token to all scene records."""
        for scene in self.scenes:
            scene["log_token"] = getattr(self, "log_token", "")

    def _generate_instance_entries(self, scene_folder: str, scene_token: str):
        """Generate instance.json entries for a scene based on 3dbbox annotations.
        
        Args:
            scene_folder: Name of the scene folder
            scene_token: Token of the scene
        """
        # Get category token mapping
        category_mappings = self.config.get("category_mappings", {})
        category_name_to_token = {entry["name"]: entry["token"] for entry in self.categories}
        
        # Track unique actor IDs in this scene
        unique_actors = set()
        
        # Find all 3dbbox JSON files in the scene
        scene_path = self.input_base / scene_folder
        for sensor_folder in scene_path.iterdir():
            if not sensor_folder.is_dir():
                continue
                
            # Look for 3dbbox JSON files
            bbox_files = list(sensor_folder.glob("*_3dbbox.json"))
            for bbox_file in bbox_files:
                try:
                    if bbox_file.stat().st_size <= 2:  # Skip empty files
                        continue
                        
                    with open(bbox_file, 'r') as f:
                        bbox_data = json.load(f)
                        
                    # Extract unique actor IDs and their types
                    for annotation in bbox_data:
                        actor_id = annotation.get("actor_id")
                        actor_type = annotation.get("type")
                        
                        if actor_id is not None and actor_type is not None:
                            unique_actors.add((actor_id, actor_type))
                            
                except Exception as e:
                    print(f"Error reading bbox file {bbox_file}: {e}")
                    continue
        
        print(f"Found {len(unique_actors)} unique actors in scene {scene_folder}")
        
        # Generate instance entries for each unique actor
        for actor_id, actor_type in unique_actors:
            # Get NuScenes category name from mapping
            nuscene_category = category_mappings.get(actor_type)
            if not nuscene_category:
                print(f"Warning: No category mapping found for CARLA type {actor_type}")
                continue
                
            # Get category token
            category_token = category_name_to_token.get(nuscene_category)
            if not category_token:
                print(f"Warning: No category token found for NuScenes category {nuscene_category}")
                continue
            
            # Generate instance entry
            instance_token = self._generate_token()
            instance_entry = {
                "token": instance_token,
                "category_token": category_token,
                "nbr_annotations": None,  # TODO: Will be populated later
                "first_annotation_token": None,  # TODO: Will be populated later
                "last_annotation_token": None  # TODO: Will be populated later
            }
            
            # Store the instance token in the mapping
            self.token_maps['instance'][(scene_token, actor_id)] = instance_token
            self.instances.append(instance_entry)

    def _generate_attribute_entries(self):
        """Generate attribute.json entries based on config."""
        print("\nGenerating attribute entries...")
        attribute_definitions = self.config.get("attributes", {})
        self.attributes = []
        
        # Process vehicle attributes
        if "vehicle" in attribute_definitions:
            for attr in attribute_definitions["vehicle"]:
                token = self._generate_token()
                attribute_entry = {
                    "token": token,
                    "name": attr["name"],
                    "description": f"Vehicle state: {attr['name'].split('.')[-1]}"
                }
                self.attributes.append(attribute_entry)
                # Store token for later use
                self.attribute_name_to_token[attr["name"]] = token
        
        print(f"Generated {len(self.attributes)} attribute entries")

    def _infer_vehicle_state(self, velocity_magnitude: float) -> str:
        """Infer vehicle state based on velocity magnitude.
        
        Args:
            velocity_magnitude: The magnitude of the vehicle's velocity.
            
        Returns:
            The name of the attribute that best describes the vehicle's state.
        """
        # Define velocity thresholds (in m/s)
        MOVING_THRESHOLD = 0.5  # Vehicles moving faster than this are considered "moving"
        
        if velocity_magnitude > MOVING_THRESHOLD:
            return "vehicle.moving"
        else:
            return "vehicle.stopped"

    def _get_sensor_transform(self, scene_folder, sensor_name, timestamp):
        """Get the transform of a sensor at a specific timestamp.
        
        Args:
            scene_folder: Name of the scene folder
            sensor_name: Name of the sensor
            timestamp: Timestamp to get transform for
            
        Returns:
            Tuple of (sensor_transform, ego_transform) or (None, None) if not found
        """
        # Load sensor config
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)
            
        # Find sensor config
        sensor_config = None
        for sensor in config["sensors"]:
            if sensor["name"] == sensor_name:
                sensor_config = sensor
                break
                
        if sensor_config is None:
            return None, None
            
        # Get ego vehicle transform at this timestamp
        ego_pose_file = self.input_base / scene_folder / "ego_pose" / f"{timestamp}.json"
        if not ego_pose_file.exists():
            return None, None
            
        try:
            with open(ego_pose_file, 'r') as f:
                ego_pose = json.load(f)
                
            # Create ego transform
            ego_loc = ego_pose["translation"]
            ego_rot = ego_pose["rotation"]
            ego_transform = carla.Transform(
                carla.Location(x=ego_loc["x"], y=ego_loc["y"], z=ego_loc["z"]),
                carla.Rotation(pitch=ego_rot["pitch"], yaw=ego_rot["yaw"], roll=ego_rot["roll"])
            )
            
            # Create sensor transform relative to ego
            loc = sensor_config["transform"]["location"]
            rot = sensor_config["transform"]["rotation"]
            sensor_transform = carla.Transform(
                carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
                carla.Rotation(pitch=rot.get("pitch", 0), yaw=rot["yaw"], roll=rot.get("roll", 0))
            )
            
            return sensor_transform, ego_transform
            
        except Exception as e:
            print(f"Warning: Error loading transforms for {sensor_name} at {timestamp}: {e}")
            return None, None

    def _is_point_in_box(self, point, box_center, box_size, box_rotation):
        """Check if a point is inside an oriented 3D box.
        
        Args:
            point: [x, y, z] point in global frame
            box_center: [x, y, z] center of box in global frame
            box_size: [width, length, height] of box
            box_rotation: quaternion [w, x, y, z] of box rotation
            
        Returns:
            bool: True if point is inside box
        """
        # Print debug info for first point
        if not hasattr(self, '_debug_point_printed'):
            print(f"\nDebug - Box parameters:")
            print(f"Box center: {box_center}")
            print(f"Box size: {box_size}")
            print(f"Box rotation (quaternion): {box_rotation}")
            print(f"Point to test: {point}")
            self._debug_point_printed = True
        
        # Convert point to box-local coordinates
        point_local = point - np.array(box_center)
        
        # Create rotation matrix from quaternion
        qw, qx, qy, qz = box_rotation
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])
        
        # Transform point to box-local frame
        point_local = np.dot(R, point_local)
        
        # Convert box size from [width, length, height] to [x, y, z] dimensions
        # In NuScenes: width = y, length = x, height = z
        box_dims = np.array([box_size[1], box_size[0], box_size[2]])  # [length, width, height]
        half_size = box_dims / 2
        
        # Print debug info for first point
        if not hasattr(self, '_debug_local_printed'):
            print(f"\nDebug - Local coordinates:")
            print(f"Point in local frame: {point_local}")
            print(f"Box half dimensions: {half_size}")
            self._debug_local_printed = True
        
        # Check if point is inside box bounds with a small tolerance
        tolerance = 0.1  # Increased tolerance to 10cm
        is_inside = all(abs(point_local) <= half_size + tolerance)
        
        # Print debug info for first point
        if not hasattr(self, '_debug_result_printed'):
            print(f"\nDebug - Point-in-box test:")
            print(f"X check: {abs(point_local[0])} <= {half_size[0] + tolerance}")
            print(f"Y check: {abs(point_local[1])} <= {half_size[1] + tolerance}")
            print(f"Z check: {abs(point_local[2])} <= {half_size[2] + tolerance}")
            print(f"Final result: {is_inside}")
            self._debug_result_printed = True
        
        return is_inside

    def _count_points_in_box(self, points, box_center, box_size, box_rotation):
        """Count how many points fall inside a 3D box.
        
        Args:
            points: numpy array of points [N x 3] or [N x 4] (with intensity)
            box_center: [x, y, z] center of box in global frame
            box_size: [width, length, height] of box
            box_rotation: quaternion [w, x, y, z] of box rotation
            
        Returns:
            int: Number of points inside box
        """
        # Reset debug flags for new box
        if hasattr(self, '_debug_point_printed'):
            delattr(self, '_debug_point_printed')
        if hasattr(self, '_debug_local_printed'):
            delattr(self, '_debug_local_printed')
        if hasattr(self, '_debug_result_printed'):
            delattr(self, '_debug_result_printed')
        
        # Handle points with or without intensity
        point_positions = points[:, :3]
        
        # Count points inside box
        count = 0
        for point in point_positions:
            if self._is_point_in_box(point, box_center, box_size, box_rotation):
                count += 1
        
        # Print summary for this box
        print(f"\nBox summary:")
        print(f"Total points tested: {len(point_positions)}")
        print(f"Points inside box: {count}")
        
        return count

    def _transform_points_to_global(self, points, sensor_transform, ego_transform):
        """Transform points from sensor frame to global frame.
        
        Args:
            points: numpy array of points [N x 3] or [N x 4] (with intensity)
            sensor_transform: CARLA transform of the sensor relative to ego
            ego_transform: CARLA transform of the ego vehicle
            
        Returns:
            Transformed points in global frame
        """
        # Get transform matrices
        sensor_matrix = np.array(sensor_transform.get_matrix())
        ego_matrix = np.array(ego_transform.get_matrix())
        
        # Handle points with or without intensity
        has_intensity = points.shape[1] == 4
        point_positions = points[:, :3]
        
        # Add homogeneous coordinate
        point_positions_h = np.column_stack((point_positions, np.ones(len(points))))
        
        # Transform points: sensor -> ego -> global
        points_ego = np.dot(point_positions_h, sensor_matrix.T)
        points_global = np.dot(points_ego, ego_matrix.T)
        
        # Combine with intensity if present
        if has_intensity:
            return np.column_stack((points_global[:, :3], points[:, 3]))
        return points_global[:, :3]

    def _transform_radar_points_to_global(self, radar_points, sensor_transform, ego_transform):
        """Transform radar points from spherical to global frame.
        
        Args:
            radar_points: numpy array of radar points [N x 5] (depth, elevation, azimuth, velocity, intensity)
            sensor_transform: CARLA transform of the sensor relative to ego
            ego_transform: CARLA transform of the ego vehicle
            
        Returns:
            Transformed points in global frame [N x 3]
        """
        # Validate input
        if radar_points.size == 0:
            print("Warning: Empty radar points array")
            return np.array([])
        
        if radar_points.shape[1] != 5:
            print(f"Warning: Unexpected radar points shape {radar_points.shape}, expected (N, 5)")
            return np.array([])
        
        # Print some debug info about the radar points
        print(f"Processing {len(radar_points)} radar points")
        print(f"Depth range: [{np.min(radar_points[:, 0]):.2f}, {np.max(radar_points[:, 0]):.2f}]")
        print(f"Elevation range: [{np.min(radar_points[:, 1]):.2f}, {np.max(radar_points[:, 1]):.2f}]")
        print(f"Azimuth range: [{np.min(radar_points[:, 2]):.2f}, {np.max(radar_points[:, 2]):.2f}]")
        
        # Convert spherical to Cartesian in sensor frame
        points = []
        for point in radar_points:
            depth, elevation, azimuth, velocity, intensity = point
            
            # Validate angles
            if not (-90 <= elevation <= 90):
                print(f"Warning: Invalid elevation angle {elevation}")
                continue
            if not (-180 <= azimuth <= 180):
                print(f"Warning: Invalid azimuth angle {azimuth}")
                continue
            if depth <= 0:
                print(f"Warning: Invalid depth {depth}")
                continue
            
            # Convert to radians
            elevation_rad = math.radians(elevation)
            azimuth_rad = math.radians(azimuth)
            
            # Convert to Cartesian
            x = depth * math.cos(elevation_rad) * math.cos(azimuth_rad)
            y = depth * math.cos(elevation_rad) * math.sin(azimuth_rad)
            z = depth * math.sin(elevation_rad)
            
            points.append([x, y, z])
        
        if not points:
            print("Warning: No valid radar points after conversion")
            return np.array([])
        
        points = np.array(points)
        
        # Transform to global frame
        global_points = self._transform_points_to_global(points, sensor_transform, ego_transform)
        
        # Print some debug info about the transformed points
        if len(global_points) > 0:
            print(f"Transformed {len(global_points)} points to global frame")
            print(f"Global points range: X[{np.min(global_points[:, 0]):.2f}, {np.max(global_points[:, 0]):.2f}], "
                  f"Y[{np.min(global_points[:, 1]):.2f}, {np.max(global_points[:, 1]):.2f}], "
                  f"Z[{np.min(global_points[:, 2]):.2f}, {np.max(global_points[:, 2]):.2f}]")
        
        return global_points

    def _generate_sample_annotations(self, scene_folder: str, scene_token: str):
        """Generate sample_annotation.json entries for a scene based on 3dbbox data.
        
        Args:
            scene_folder: Name of the scene folder
            scene_token: Token of the scene
        """
        print(f"\nGenerating sample annotations for scene {scene_folder}")
        
        # Build mappings for linking - only use samples from this scene
        sample_token_map = {entry["timestamp"]: entry["token"] for entry in self.samples if entry["scene_token"] == scene_token}
        print(f"Found {len(sample_token_map)} samples to process")
        print(f"Sample timestamps: {sorted(sample_token_map.keys())}")
        
        # Track annotations per instance for next/prev linking
        instance_annotations = {}  # instance_token -> list of (timestamp, annotation_token)
        
        # Find all 3dbbox JSON files in the scene
        scene_path = self.input_base / scene_folder
        bbox_files_found = 0
        empty_files = 0
        annotations_processed = 0
        skipped_timestamps = 0
        
        for sensor_folder in scene_path.iterdir():
            if not sensor_folder.is_dir():
                continue
                
            print(f"\nProcessing sensor folder: {sensor_folder.name}")
            
            # Look for 3dbbox JSON files
            bbox_files = list(sensor_folder.glob("*_3dbbox.json"))
            bbox_files_found += len(bbox_files)
            
            for bbox_file in bbox_files:
                try:
                    # Skip empty files (2 bytes or less)
                    if bbox_file.stat().st_size <= 2:
                        empty_files += 1
                        continue
                        
                    # Extract timestamp from filename
                    timestamp = int(bbox_file.stem.split('_')[0])
                    sample_token = sample_token_map.get(timestamp)
                    if not sample_token:
                        skipped_timestamps += 1
                        continue
                        
                    with open(bbox_file, 'r') as f:
                        bbox_data = json.load(f)
                        
                    if not bbox_data:  # Skip if file is empty list
                        empty_files += 1
                        continue
                        
                    # Process each annotation in the file
                    for annotation in bbox_data:
                        actor_id = annotation.get("actor_id")
                        if actor_id is None:
                            print(f"Warning: Missing actor_id in annotation from {bbox_file}")
                            continue
                            
                        # Get instance token from mapping
                        instance_token = self.token_maps['instance'].get((scene_token, actor_id))
                        if not instance_token:
                            print(f"Warning: No instance token found for actor {actor_id}")
                            continue
                            
                        # Extract pose data
                        pose = annotation.get("pose", {})
                        translation = pose.get("translation", {})
                        rotation = pose.get("rotation", {})
                        
                        # Convert rotation from Euler to Quaternion
                        quaternion = self._euler_to_quaternion(
                            float(rotation.get("roll", 0)),
                            float(rotation.get("pitch", 0)),
                            float(rotation.get("yaw", 0))
                        )
                        
                        # Get velocity and infer vehicle state
                        velocity = annotation.get("velocity", {})
                        velocity_magnitude = velocity.get("magnitude", 0.0)
                        vehicle_state = self._infer_vehicle_state(velocity_magnitude)
                        attribute_token = self.attribute_name_to_token.get(vehicle_state, "")
                        
                        # Get box parameters
                        box_center = [
                            float(translation.get("x", 0)),
                            float(translation.get("y", 0)),
                            float(translation.get("z", 0))
                        ]
                        box_size = annotation.get("size", [0, 0, 0])
                        
                        # Count lidar points
                        num_lidar_pts = 0
                        lidar_file = scene_path / "Lidar" / f"{timestamp}.npy"
                        if lidar_file.exists():
                            try:
                                # Load lidar points (x, y, z, intensity)
                                lidar_points = np.load(lidar_file)
                                
                                # Get transforms
                                sensor_transform, ego_transform = self._get_sensor_transform(scene_folder, "Lidar", timestamp)
                                if sensor_transform is not None and ego_transform is not None:
                                    # Transform points to global frame
                                    global_points = self._transform_points_to_global(lidar_points, sensor_transform, ego_transform)
                                    
                                    # Count points in box
                                    num_lidar_pts = self._count_points_in_box(global_points, box_center, box_size, quaternion)
                            except Exception as e:
                                print(f"Warning: Error processing lidar points: {e}")
                        
                        # Count radar points
                        num_radar_pts = 0
                        for radar_name in ["Radar_Front", "Radar_Back", "Radar_FrontRight", "Radar_FrontLeft", "Radar_BackRight", "Radar_BackLeft"]:
                            radar_file = scene_path / radar_name / f"{timestamp}.npy"
                            if radar_file.exists():
                                try:
                                    # Load radar points (depth, elevation, azimuth, velocity, intensity)
                                    radar_points = np.load(radar_file)
                                    
                                    # Get transforms
                                    sensor_transform, ego_transform = self._get_sensor_transform(scene_folder, radar_name, timestamp)
                                    if sensor_transform is not None and ego_transform is not None:
                                        # Transform points to global frame
                                        global_points = self._transform_radar_points_to_global(radar_points, sensor_transform, ego_transform)
                                        
                                        if len(global_points) > 0:
                                            # Count points in box
                                            box_count = self._count_points_in_box(global_points, box_center, box_size, quaternion)
                                            num_radar_pts += box_count
                                            print(f"Radar {radar_name}: Found {box_count} points in box")
                                except Exception as e:
                                    print(f"Warning: Error processing radar points for {radar_name}: {e}")
                        
                        # Create annotation entry
                        annotation_token = self._generate_token()
                        annotation_entry = {
                            "token": annotation_token,
                            "sample_token": sample_token,
                            "instance_token": instance_token,
                            "attribute_tokens": [attribute_token] if attribute_token else [],
                            "visibility_token": "",  # TODO: Will be populated later
                            "translation": box_center,
                            "size": box_size,
                            "rotation": quaternion,
                            "num_lidar_pts": num_lidar_pts,
                            "num_radar_pts": num_radar_pts,
                            "next": "",  # Will be populated after all annotations are processed
                            "prev": ""   # Will be populated after all annotations are processed
                        }
                        
                        # Add to instance tracking
                        if instance_token not in instance_annotations:
                            instance_annotations[instance_token] = []
                        instance_annotations[instance_token].append((timestamp, annotation_token))
                        
                        self.sample_annotations.append(annotation_entry)
                        annotations_processed += 1
                        
                except Exception as e:
                    print(f"Error processing bbox file {bbox_file}: {e}")
                    continue
        
        print(f"\nSummary for scene {scene_folder}:")
        print(f"Total bbox files found: {bbox_files_found}")
        print(f"Empty/invalid files: {empty_files}")
        print(f"Skipped timestamps (no sample token): {skipped_timestamps}")
        print(f"Successfully processed annotations: {annotations_processed}")
        print(f"Unique instances with annotations: {len(instance_annotations)}")
        
        # Link next/prev tokens for each instance
        for instance_token, annotations in instance_annotations.items():
            # Sort by timestamp
            annotations.sort(key=lambda x: x[0])
            
            # Link annotations
            for i, (_, annotation_token) in enumerate(annotations):
                annotation = next(a for a in self.sample_annotations if a["token"] == annotation_token)
                
                if i > 0:
                    prev_token = annotations[i-1][1]
                    annotation["prev"] = prev_token
                    
                if i < len(annotations) - 1:
                    next_token = annotations[i+1][1]
                    annotation["next"] = next_token
        
        if not self.sample_annotations:
            print("Warning: No sample annotations were generated!")
        else:
            print(f"Successfully generated {len(self.sample_annotations)} sample annotations")

    def convert_scene(self, scene_folder: str):
        """Convert a single scene folder to NuScenes format.

        Args:
            scene_folder: Name of the scene folder to convert.
        """
        # Parse data files
        data_files = self._find_data_files(scene_folder)
        if not data_files:
             print(f"Warning: No data files found in {scene_folder}. Skipping.")
             return

        # Organize data by timestamp
        organized_data = self._organize_data_by_timestamp(data_files)
        if not organized_data:
             print(f"Warning: Could not organize data by timestamp in {scene_folder}. Skipping.")
             return

        # Generate scene token
        scene_token = self._generate_token()
        self.token_maps['scene'][scene_folder] = scene_token

        # Generate instance entries for this scene
        self._generate_instance_entries(scene_folder, scene_token)

        # Select keyframes first
        target_rate = self.config['output']['keyframe_rate']
        keyframes = self._select_keyframes(list(organized_data.keys()), target_rate)
        if not keyframes:
             print(f"Warning: No keyframes selected for scene {scene_folder}. Skipping sample generation.")
             return

        # Generate sample entries for this scene
        scene_samples = self._generate_sample_entries(keyframes, scene_token)
        self.samples.extend(scene_samples)

        # Generate sample annotations after samples are created
        self._generate_sample_annotations(scene_folder, scene_token)

        # --- Create the scene record ---
        self.scenes.append({
            "token": scene_token,
            "log_token": "",  # Assign log token later if needed
            "name": scene_folder,
            "description": f"Scene data for {scene_folder}",
            "nbr_samples": len(scene_samples),
            "first_sample_token": scene_samples[0]["token"] if scene_samples else "",
            "last_sample_token": scene_samples[-1]["token"] if scene_samples else "",
        })

        # After generating samples and calibrated sensors, generate sample_data
        self._generate_sample_data_entries(scene_folder, scene_token)

    def _generate_category_entries(self):
        """Generate category.json entries based on category mappings in config."""
        category_mappings = self.config.get("category_mappings", {})
        self.categories = []
        
        for carla_type, nuscene_name in category_mappings.items():
            token = self._generate_token()
            description = "A car" if nuscene_name == "vehicle.car" else ""
            # TODO: Add more descriptions for other categories
            
            category_entry = {
                "token": token,
                "name": nuscene_name,
                "description": description,
                "index": None  # TODO later
            }
            self.categories.append(category_entry)

    def _update_instance_entries(self):
        """Update instance entries with annotation counts and first/last annotation tokens."""
        print("\nUpdating instance entries with annotation information...")
        
        # Create a mapping of sample tokens to timestamps
        sample_token_to_timestamp = {sample["token"]: sample["timestamp"] for sample in self.samples}
        
        # Group annotations by instance token
        instance_annotations = {}
        for annotation in self.sample_annotations:
            instance_token = annotation["instance_token"]
            if instance_token not in instance_annotations:
                instance_annotations[instance_token] = []
            instance_annotations[instance_token].append(annotation)
        
        # Update each instance entry
        for instance in self.instances:
            instance_token = instance["token"]
            annotations = instance_annotations.get(instance_token, [])
            
            if annotations:
                # Sort annotations by timestamp using the sample token mapping
                annotations.sort(key=lambda x: sample_token_to_timestamp.get(x["sample_token"], 0))
                
                # Update instance entry
                instance["nbr_annotations"] = len(annotations)
                instance["first_annotation_token"] = annotations[0]["token"]
                instance["last_annotation_token"] = annotations[-1]["token"]
            else:
                # No annotations found for this instance
                instance["nbr_annotations"] = 0
                instance["first_annotation_token"] = ""
                instance["last_annotation_token"] = ""
        
        print(f"Updated {len(self.instances)} instance entries")
        print(f"Instances with annotations: {sum(1 for i in self.instances if i['nbr_annotations'] > 0)}")

    def convert_all(self):
        """Convert all scenes specified in the config."""
        self._generate_sensor_entries()
        self._generate_calibrated_sensors()
        self._generate_log_entry()
        self._generate_category_entries()
        self._generate_attribute_entries()  # Generate attributes before processing scenes
        for scene_folder in self.scene_folders:
            print(f"Processing scene: {scene_folder}")
            self.convert_scene(scene_folder)
        self._assign_log_token_to_scenes()
        # Update instance entries after all scenes are processed
        self._update_instance_entries()
        self._write_all_tables()

    def _write_metadata(self):
        """Write all metadata files in NuScenes format."""
        metadata_files = {
            'attribute': self.attributes,
            'calibrated_sensor': self.calibrated_sensors,
            'category': self.categories,
            'ego_pose': self.ego_poses,
            'instance': self.instances,
            'log': self.logs,
            'map': self.maps,
            'sample': self.samples,
            'sample_annotation': self.sample_annotations,
            'sample_data': self.sample_data,
            'scene': self.scenes,
            'sensor': self.sensors,
            'visibility': self.visibilities,
        }

        for name, data in metadata_files.items():
            output_path = self.output_base / f'{name}.json'
            if name == 'sample':
                print(f"DEBUG: Writing {len(self.samples)} samples to sample.json")  # Debugging output before writing samples
                print(f"DEBUG: Sample data before writing: {data}")  # Debugging output to verify sample data before writing
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

    def _write_output(self, table_name: str, data: List[dict]):
        """Write a specific NuScenes table to a JSON file.

        Args:
            table_name: The name of the NuScenes table (e.g., 'scene', 'sample').
            data: The list of records to write to the file.
        """
        output_path = self.output_base / f"{table_name}.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _write_all_tables(self):
        """Write all NuScenes tables to their respective JSON files."""
        tables = {
            'attribute': self.attributes,
            'calibrated_sensor': self.calibrated_sensors,
            'category': self.categories,
            'ego_pose': self.ego_poses,
            'instance': self.instances,
            'log': self.logs,
            'map': self.maps,
            'sample': self.samples,
            'sample_annotation': self.sample_annotations,
            'sample_data': self.sample_data,
            'scene': self.scenes,
            'sensor': self.sensors,  # Use the self.sensors populated earlier
            'visibility': self.visibilities,
        }

        for table_name, data in tables.items():
            # Skip writing calibrated_sensor and sensor here, handle them explicitly below
            if table_name not in ['calibrated_sensor', 'sensor']:
                 self._write_output(table_name, data)

        # Write calibrated_sensor.json
        output_path = self.output_base / 'calibrated_sensor.json'
        with open(output_path, 'w') as f:
            json.dump(self.calibrated_sensors, f, indent=2)

        # Write sensor.json explicitly to ensure it contains the tokens used by calibrated_sensor
        sensor_output_path = self.output_base / 'sensor.json'
        with open(sensor_output_path, 'w') as f:
            json.dump(self.sensors, f, indent=2)

def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python carla_to_nuscene_converter.py path/to/converter_config.yml")
        sys.exit(1)
        
    config_path = sys.argv[1]
    converter = NuScenesConverter(config_path)
    converter.convert_all()

if __name__ == '__main__':
    main()
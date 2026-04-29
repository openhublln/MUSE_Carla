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
import shutil
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent  # MUSE_Carla/
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import utilities from the new module
from nuscene_utils import (
    generate_token,
    generate_composite_token,
    euler_to_quaternion,
    convert_bounding_box_size,
    transform_to_global_frame,
    adjust_z_for_ego_pose,
    is_point_in_box,
    transform_points_to_global,
    transform_radar_points_to_global,
    count_points_in_box
)

# Import new generator class
from sensor_calibrated_generators import SensorCalibratedGenerators
from log_generator import LogGenerator
from metadata_generators import MetadataGenerators
from instance_generator import InstanceGenerator
from sample_generator import SampleGenerator
from sample_data_generator import SampleDataGenerator
from annotation_generator import AnnotationGenerator

# Import the comprehensive fix module
from nuscenes_fixes import NuScenesFixes

class NuScenesConverter:
    """Converts CARLA sensor data to NuScenes format."""
    
    EGO_POSE_FOLDER = "ego_pose" # Class constant

    def _detect_scene_folders(self) -> List[str]:
        """Detect scene folders in the input directory.
        
        Returns:
            List of scene folder names
        """
        print(f"Searching for scene folders in: {self.input_base.absolute()}")
        
        # Look for folders that match the scene pattern
        scene_folders = []
        for item in self.input_base.iterdir():
            if item.is_dir() and item.name.startswith('scene_'):
                print(f"Found scene folder: {item.name}")
                scene_folders.append(item.name)
                
        if not scene_folders:
            print("WARNING: No scene folders found!")
            print(f"Contents of {self.input_base.absolute()}:")
            for item in self.input_base.iterdir():
                print(f"  - {item.name} ({'directory' if item.is_dir() else 'file'})")
        else:
            print(f"Found {len(scene_folders)} scene folders")
            
        return sorted(scene_folders)

    def __init__(self, config_path: str):
        """Initialize the converter with the given config file.
        
        Args:
            config_path: Path to the converter_config.yml file
        """
        self.config = self._load_config(config_path)
        self.version = self.config['version']
        
        # Set input base from config (fallback: _out)
        self.input_base = Path(self.config.get('input', {}).get('base_dir', '_out'))
        if not self.input_base.exists():
            raise FileNotFoundError(f"Input directory not found at {self.input_base.absolute()}")
            
        # Set output base from config (fallback: nuscenes_format)
        self.output_base = Path(self.config.get('output', {}).get('base_dir', 'nuscenes_format'))
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        print(f"Input base directory: {self.input_base.absolute()}")
        print(f"Output base directory: {self.output_base.absolute()}")
        
        self.scene_folders = self._detect_scene_folders()  # Automatically detect scenes
        print(f"Detected scene folders: {self.scene_folders}")

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
        
        # Debug flags for coordinate transformations
        self._debug_calibrations = False  # Set to True to enable debug output
        self._debug_annotations = False  # Disable debug output for cleaner conversion

        # Visibility computation cache: (scene_folder, timestamp, actor_id) -> avg_visibility
        self._visibility_cache: Dict[Tuple[str, int, int], float] = {}

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
        # Add a default map entry to prevent reverse indexing errors
        self.maps = [{
            "token": self._generate_token(),
            "log_tokens": [],  # Will be populated with log tokens
            "category": "semantic_prior",
            "filename": "maps/none.png"
        }]
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
            'visibility': {},  # (scene_folder, actor_id) -> token
            'log': {},  # scene_name -> token
            'ego_pose': {},  # (scene_folder, timestamp) -> ego_pose_token
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
            self.token_maps['instance'][key] = generate_composite_token(scene_id, actor_id, sensor_name)
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
            # Try to find it relative to MUSE_Carla root
            sim_config_path = ROOT / 'config.yml'
            if not sim_config_path.exists():
                 # As a last resort, assume it's just 'config.yml' in the current working directory
                sim_config_path = Path('config.yml')
                if not sim_config_path.exists():
                    raise FileNotFoundError("Could not find simulation config.yml")

        with open(sim_config_path, 'r') as f:
            self.sim_config = yaml.safe_load(f)

        # Load epoch base from log_info.json written at collection time
        log_info_path = self.input_base / 'log_info.json'
        if log_info_path.exists():
            with open(log_info_path, 'r') as f:
                log_info = json.load(f)
            self.epoch_base_us = int(log_info.get('start_time_unix', 0) * 1_000_000)
        else:
            self.epoch_base_us = 0

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
        return transform_to_global_frame(transform, self.global_origin)

    def _adjust_z_for_ego_pose(self, translation):
        """Adjust the Z-coordinate for ego_pose to match NuScenes' Z=0 assumption.

        Args:
            translation: A list representing the [x, y, z] translation.

        Returns:
            A modified translation with Z set to 0.
        """
        return adjust_z_for_ego_pose(translation)

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
        if not hasattr(self, 'sample_gen_instance') or self.sample_gen_instance is None:
            self.sample_gen_instance = SampleGenerator(self)
        return self.sample_gen_instance.select_keyframes(timestamps, target_rate)

    def _generate_sample_entries(self, keyframes: List[int], scene_token: str) -> List[dict]:
        if not hasattr(self, 'sample_gen_instance') or self.sample_gen_instance is None:
            self.sample_gen_instance = SampleGenerator(self)
        return self.sample_gen_instance.generate_sample_entries(keyframes, scene_token)

    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> List[float]:
        """Convert Euler angles (roll, pitch, yaw) to a Quaternion (w, x, y, z).

        Args:
            roll: Rotation around the X-axis in degrees.
            pitch: Rotation around the Y-axis in degrees.
            yaw: Rotation around the Z-axis in degrees.

        Returns:
            A list representing the Quaternion [w, x, y, z].
        """
        from .nuscene_utils import carla_rotation_to_nuscenes_quaternion
        return carla_rotation_to_nuscenes_quaternion(roll, pitch, yaw)

    def _convert_bounding_box_size(self, extent) -> List[float]:
        """Convert CARLA bounding box extent to NuScenes size.

        Args:
            extent: A CARLA bounding box extent object with x, y, z (half-dimensions).

        Returns:
            A list representing the size [width, length, height] in NuScenes format.
        """
        return convert_bounding_box_size(extent)

    def _generate_sensor_entries(self):
        """Generate sensor entries with correct channel names."""
        sensor_channels = {
            'Camera_Front': 'CAM_FRONT',
            'Camera_Back': 'CAM_BACK',
            'Camera_FrontRight': 'CAM_FRONT_RIGHT',
            'Camera_FrontLeft': 'CAM_FRONT_LEFT',
            'Camera_BackRight': 'CAM_BACK_RIGHT',
            'Camera_BackLeft': 'CAM_BACK_LEFT',
            'Lidar': 'LIDAR_TOP', 
            'Radar_Front': 'RADAR_FRONT',
            'Radar_FrontRight': 'RADAR_FRONT_RIGHT',
            'Radar_FrontLeft': 'RADAR_FRONT_LEFT',
            'Radar_BackRight': 'RADAR_BACK_RIGHT',
            'Radar_BackLeft': 'RADAR_BACK_LEFT',
            'GNSS': 'GNSS',
            'IMU': 'IMU'
        }
        
        for sensor_name, channel in sensor_channels.items():
            modality = 'camera' if sensor_name.startswith('Camera') else \
                      'lidar' if sensor_name == 'Lidar' else \
                      'semantic_lidar' if sensor_name == 'Semantic_Lidar' else \
                      'radar' if sensor_name.startswith('Radar') else \
                      sensor_name.lower()
            
            self.sensors.append({
                "token": self._generate_token(),
                "channel": channel,
                "modality": modality
            })
            # Store the token for later use
            self.token_maps['calibrated_sensor'][sensor_name] = self.sensors[-1]['token']

    def _generate_calibrated_sensors(self):
        sc_generators = SensorCalibratedGenerators(self)
        sc_generators.generate_calibrated_sensors()

    def _generate_sample_data_entries(self, scene_folder: str, scene_token: str):
        """Generate sample data entries."""
        if not hasattr(self, 'sample_data_gen_instance') or self.sample_data_gen_instance is None:
            self.sample_data_gen_instance = SampleDataGenerator(self)
        
        # Generate sample data entries
        self.sample_data_gen_instance.generate_sample_data_entries(scene_folder, scene_token)

    def _generate_sample_annotations(self, scene_folder: str, scene_token: str):
        if not hasattr(self, 'annotation_gen_instance') or self.annotation_gen_instance is None:
            self.annotation_gen_instance = AnnotationGenerator(self)
        self.annotation_gen_instance.generate_sample_annotations(scene_folder, scene_token)

    def _get_sensor_transform(self, scene_folder: str, sensor_name: str, timestamp: int) -> Tuple[Optional[carla.Transform], Optional[carla.Transform]]:
        """Get the sensor's transform relative to ego, and ego's transform in global CARLA coordinates.
        
        Args:
            scene_folder: Name of the scene folder
            sensor_name: Name of the sensor
            timestamp: Timestamp to get transform for
            
        Returns:
            Tuple of (sensor_transform, ego_transform) or (None, None) if not found
        """
        # Load sensor config from self.sim_config
        sensor_config = None
        for sensor_cfg in self.sim_config.get("sensors", []):
            if sensor_cfg["name"] == sensor_name:
                sensor_config = sensor_cfg
                break
                
        if sensor_config is None:
            return None, None
            
        # Get ego vehicle transform at this timestamp
        ego_pose_file = self.input_base / scene_folder / self.EGO_POSE_FOLDER / f"{timestamp}.json"
        if not ego_pose_file.exists():
            return None, None
            
        try:
            with open(ego_pose_file, 'r') as f:
                ego_pose = json.load(f)
                
            ego_loc_data = ego_pose["translation"]
            ego_rot_data = ego_pose["rotation"]
            ego_transform = carla.Transform(
                carla.Location(x=float(ego_loc_data["x"]), y=float(ego_loc_data["y"]), z=float(ego_loc_data["z"])),
                carla.Rotation(pitch=float(ego_rot_data["pitch"]), yaw=float(ego_rot_data["yaw"]), roll=float(ego_rot_data["roll"])))
            
            loc_data = sensor_config["transform"]["location"]
            rot_data = sensor_config["transform"]["rotation"]
            sensor_transform_relative = carla.Transform(
                carla.Location(x=float(loc_data["x"]), y=float(loc_data["y"]), z=float(loc_data["z"])),
                carla.Rotation(pitch=float(rot_data.get("pitch", 0.0)), yaw=float(rot_data["yaw"]), roll=float(rot_data.get("roll", 0.0))))
            
            return sensor_transform_relative, ego_transform
            
        except Exception as e:
            print(f"Warning: Error loading transforms for {sensor_name} at {timestamp} in scene {scene_folder}: {e}")
            return None, None

    def _update_sample_annotations_with_visibility(self, scene_folder: str):
        if not hasattr(self, 'annotation_gen_instance') or self.annotation_gen_instance is None:
            self.annotation_gen_instance = AnnotationGenerator(self)
        self.annotation_gen_instance.update_sample_annotations_with_visibility(scene_folder)

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
        scene_token = generate_token()
        self.token_maps['scene'][scene_folder] = scene_token

        # Initialize SampleGenerator if not already done (e.g. if convert_scene is called directly)
        if not hasattr(self, 'sample_gen_instance') or self.sample_gen_instance is None:
            self.sample_gen_instance = SampleGenerator(self)

        # Select keyframes first
        target_rate = self.config['output']['keyframe_rate']
        keyframes = self.sample_gen_instance.select_keyframes(list(organized_data.keys()), target_rate)
        if not keyframes:
             print(f"Warning: No keyframes selected for scene {scene_folder}. Skipping sample generation.")
             return

        # Generate sample entries for this scene
        scene_samples = self.sample_gen_instance.generate_sample_entries(keyframes, scene_token)
        self.samples.extend(scene_samples)

        # Instance entries must be generated before sample annotations
        # as annotations rely on instance tokens.
        self.instance_gen_instance.generate_instance_entries(scene_folder, scene_token)

        # Generate sample annotations after samples and instances are created
        self._generate_sample_annotations(scene_folder, scene_token)
        
        # Update sample annotations with visibility tokens
        self._update_sample_annotations_with_visibility(scene_folder)

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
        self.sample_data_gen_instance.generate_sample_data_entries(scene_folder, scene_token)
        # Ensure every sample_data timestamp has a matching ego_pose (interpolates missing)
        self.sample_data_gen_instance.ensure_ego_poses_for_scene_sample_data(scene_folder)

    def convert_all(self):
        """Convert all scenes specified in the config."""
        # Initialize the new generator classes
        sc_generators = SensorCalibratedGenerators(self)
        sc_generators.generate_sensor_entries()
        sc_generators.generate_calibrated_sensors()

        log_gen = LogGenerator(self)
        log_gen.generate_log_entry()

        meta_gen = MetadataGenerators(self)
        meta_gen.generate_category_entries()
        meta_gen.generate_attribute_entries()
        meta_gen.generate_visibility_entries()

        # Instance generator will be used per scene for generation 
        # and once at the end for updates.
        self.instance_gen_instance = InstanceGenerator(self) 
        self.sample_gen_instance = SampleGenerator(self) # Initialize SampleGenerator
        self.sample_data_gen_instance = SampleDataGenerator(self) # Initialize SampleDataGenerator
        self.annotation_gen_instance = AnnotationGenerator(self) # Initialize AnnotationGenerator

        # Generate ego poses for all scenes first (before any annotation generation)
        print("Generating ego poses for all scenes...")
        for scene_folder in self.scene_folders:
            self.sample_data_gen_instance.generate_ego_poses_for_scene(scene_folder)
        print(f"Generated {len(self.ego_poses)} ego poses total")

        for scene_folder in self.scene_folders:
            print(f"Processing scene: {scene_folder}")
            self.convert_scene(scene_folder)
        
        log_gen.assign_log_token_to_scenes()
        self.instance_gen_instance.update_instance_entries()
        self._write_all_tables()
        
        # Apply comprehensive nuScenes format fixes
        print("\n" + "="*50)
        print("APPLYING AUTOMATIC NUSCENES FORMAT FIXES")
        print("="*50)
        fixes = NuScenesFixes(self.output_base, self.version)
        fixes.fix_all_issues()
        print("="*50)
        print("CONVERSION COMPLETED SUCCESSFULLY!")
        print("="*50)

    def _write_output(self, table_name: str, data: List[dict]):
        """Write a specific NuScenes table to a JSON file.

        Args:
            table_name: The name of the NuScenes table (e.g., 'scene', 'sample').
            data: The list of records to write to the file.
        """
        # Create version directory if it doesn't exist
        version_dir = self.output_base / self.version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Write JSON files ONLY to version directory
        output_path = version_dir / f"{table_name}.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Writing {table_name}.json to {output_path}")

    def _setup_map_files(self):
        """
        Finds generated map files in the input directory, copies them to the correct
        'maps/' structure under the output directory, and updates the internal map record.
        """
        source_map_dir = self.input_base
        dest_maps_dir = self.output_base / 'maps'

        # Set default values for the map record.
        map_category = 'semantic_prior'
        map_filename = 'maps/none.png'

        try:
            # List all files in _out for debugging
            all_files = list(source_map_dir.glob('*'))
            
            # Find the map metadata JSON file in the source directory.
            source_map_json_files = [f for f in source_map_dir.glob('*.json') if f.stem not in ['log_info', 'config']]
            
            if not source_map_json_files:
                print("Warning: No map metadata .json found in '_out/'.")
                raise FileNotFoundError("No map JSON found")

            source_json_path = source_map_json_files[0]
            
            with open(source_json_path, 'r') as f:
                map_meta = json.load(f)
            
            # Use the original map name to preserve the CARLA map identity
            original_map_name = map_meta.get('original_carla_map', source_json_path.stem)
            mask_filename_stem = map_meta.get('basemap', {}).get('filename', f"{original_map_name}_basemap.png")
            source_mask_path = source_map_dir / mask_filename_stem
            
            if not source_mask_path.exists():
                print(f"Warning: Found map metadata at {source_json_path} but mask '{mask_filename_stem}' is missing in '_out/'.")
                raise FileNotFoundError(f"Map mask {mask_filename_stem} not found")

            # Define destination paths using original map name
            dest_json_path = dest_maps_dir / f"{original_map_name}.json"
            dest_map_sub_dir = dest_maps_dir / original_map_name
            # Keep original filename structure
            dest_mask_path = dest_map_sub_dir / f"{original_map_name}_basemap.png"
            dest_map_sub_dir.mkdir(parents=True, exist_ok=True)

            # Copy files and update map record variables
            shutil.copy(source_mask_path, dest_mask_path)
            shutil.copy(source_json_path, dest_json_path)
            
            # Create minimal vector map for NuScenesMap API compatibility
            self._create_minimal_vector_map(dest_map_sub_dir, original_map_name, map_meta)
            
            map_category = 'semantic_prior'
            map_filename = f'maps/{original_map_name}/{original_map_name}_basemap.png'

        except (FileNotFoundError, Exception) as e:
            if isinstance(e, FileNotFoundError):
                pass # Warning already printed
            else:
                print(f"An error occurred during map file setup: {e}")
            print("Falling back to default 'none' map.")
            
            # Ensure the fallback 'none.png' exists
            mask_file = dest_maps_dir / 'none.png'
            if not mask_file.exists():
                from PIL import Image
                img = Image.new('RGB', (100, 100), color=(200, 200, 200))
                img.save(mask_file)
                print(f"Created minimal fallback map mask at {mask_file}")
        
        # Finally, update the map record in self.maps
        if self.maps:
            print(f"Before update: category='{self.maps[0]['category']}', filename='{self.maps[0]['filename']}'")
            print(f"Will update to: category='{map_category}', filename='{map_filename}'")
            self.maps[0]['category'] = map_category
            self.maps[0]['filename'] = map_filename
            print(f"After update: category='{self.maps[0]['category']}', filename='{self.maps[0]['filename']}'")
            
            # Verify the file actually exists at the expected location
            expected_file_path = self.output_base / map_filename
            if expected_file_path.exists():
                print(f"Map file created successfully at: {expected_file_path}")
            else:
                print(f"Warning: Map file not found at expected location: {expected_file_path}")
                # If file doesn't exist, fall back to default
                map_filename = 'maps/none.png'
                map_category = 'semantic_prior'
                print(f"Falling back to: {map_filename}")
                # Update the map record with fallback values
                self.maps[0]['category'] = map_category
                self.maps[0]['filename'] = map_filename
        else:
            print("Warning: No map records found to update!")

    def _create_minimal_vector_map(self, map_dir, map_name, map_meta):
        """Create minimal vector map files for NuScenesMap compatibility."""
        try:
            # Create basic map structure for NuScenesMap
            basemap_meta = {
                "origin": map_meta.get("origin", [0.0, 0.0]),
                "scale": map_meta.get("scale", 0.1),
                "rotation": map_meta.get("rotation", 0.0),
                "filename": f"{map_name}_basemap.png"  # Use the actual filename, not the original
            }
            
            # Write basemap metadata
            basemap_json_path = map_dir / f"{map_name}_basemap.json"
            with open(basemap_json_path, 'w') as f:
                json.dump(basemap_meta, f, indent=2)
            
            # Create minimal vector map files that nuScenes expects for proper map rendering
            # This is what enables underlay_map=True to work with custom maps
            
            # 1. Create main map metadata file
            map_meta_path = map_dir / f"{map_name}.json"
            main_map_meta = {
                "map_name": map_name,
                "layer_names": ["drivable_area", "lane", "ped_crossing", "walkway", "stop_line"],
                "meta": {
                    "coordinate_system": "local_meters",
                    "uses_canvas": True,
                    "canvas_edge": 50.0,
                }
            }
            with open(map_meta_path, 'w') as f:
                json.dump(main_map_meta, f, indent=2)
            
            # 2. Create empty vector layer files (required for NuScenesMap API)
            vector_layers = {
                "drivable_area": {"type": "MultiPolygon", "coordinates": []},
                "lane": {"type": "MultiLineString", "coordinates": []},
                "ped_crossing": {"type": "MultiPolygon", "coordinates": []},
                "walkway": {"type": "MultiPolygon", "coordinates": []},
                "stop_line": {"type": "MultiLineString", "coordinates": []}
            }
            
            for layer_name, geom_data in vector_layers.items():
                layer_path = map_dir / f"{map_name}_{layer_name}.json"
                with open(layer_path, 'w') as f:
                    json.dump(geom_data, f, indent=2)
            
            # 3. Create expansion metadata (for map bounds) in the correct directory structure
            # nuScenes expects expansion files in maps/expansion/ directory
            expansion_dir = self.output_base / 'maps' / 'expansion'
            expansion_dir.mkdir(parents=True, exist_ok=True)
            expansion_path = expansion_dir / f"{map_name}.json"
            
            bounds = [
                map_meta.get("origin", [0.0, 0.0])[0],  # min_x
                map_meta.get("origin", [0.0, 0.0])[1] - (basemap_meta.get("size", [1024, 1024])[1] * basemap_meta["scale"]),  # min_y  
                map_meta.get("origin", [0.0, 0.0])[0] + (basemap_meta.get("size", [1024, 1024])[0] * basemap_meta["scale"]),  # max_x
                map_meta.get("origin", [0.0, 0.0])[1]   # max_y
            ]
            expansion_meta = {
                "node_dict": {},
                "way_dict": {},
                "bounds": bounds
            }
            with open(expansion_path, 'w') as f:
                json.dump(expansion_meta, f, indent=2)
                
            print(f"Created complete vector map files for {map_name} (enables underlay_map=True)")
            
        except Exception as e:
            print(f"Warning: Failed to create vector map files: {e}")

    def _write_all_tables(self):
        """Write all NuScenes tables to their respective JSON files."""
        # Link logs to map (map.log_tokens should contain log tokens)
        for log in self.logs:
            self.maps[0]['log_tokens'].append(log['token'])

        # Create required directories
        version_dir = self.output_base / self.version
        samples_dir = self.output_base / 'samples'
        sweeps_dir = self.output_base / 'sweeps'
        maps_dir = self.output_base / 'maps'
        
        for directory in [version_dir, samples_dir, sweeps_dir, maps_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # --- Setup map files and update map record ---
        self._setup_map_files()

        # Create tables dictionary AFTER map setup to capture updated map record
        tables = {
            'attribute': self.attributes,
            'category': self.categories,
            'ego_pose': self.ego_poses,
            'instance': self.instances,
            'log': self.logs,
            'map': self.maps,
            'sample': self.samples,
            'sample_annotation': self.sample_annotations,
            'sample_data': self.sample_data,
            'scene': self.scenes,
            'visibility': self.visibilities,
            'sensor': self.sensors,
            'calibrated_sensor': self.calibrated_sensors
        }

        # Validate required tables
        required_tables = [
            'attribute', 'category', 'ego_pose', 'instance', 'log',
            'sample', 'sample_annotation', 'sample_data', 'scene',
            'visibility', 'sensor', 'calibrated_sensor'
        ]
        
        for table in required_tables:
            if not tables[table]:
                print(f"Warning: {table} table is empty")
                if table == 'map':
                    # Create empty map table as it's optional
                    tables[table] = []
                else:
                    print(f"Error: Required table {table} is empty")
                    return

        # Validate token relationships
        print("\nValidating token relationships...")
        
        # Create token lookup dictionaries
        sample_tokens = {s['token']: s for s in self.samples}
        scene_tokens = {s['token']: s for s in self.scenes}
        instance_tokens = {i['token']: i for i in self.instances}
        sensor_tokens = {s['token']: s for s in self.sensors}
        calibrated_sensor_tokens = {cs['token']: cs for cs in self.calibrated_sensors}
        log_tokens = {l['token']: l for l in self.logs}
        
        # Validate scene-sample relationships
        for scene in self.scenes:
            if scene['first_sample_token'] and scene['first_sample_token'] not in sample_tokens:
                print(f"Error: Scene {scene['token']} references non-existent sample {scene['first_sample_token']}")
                return
                
        # Validate sample-sample_data relationships
        for sample_data in self.sample_data:
            if sample_data['sample_token'] not in sample_tokens:
                print(f"Error: Sample data {sample_data['token']} references non-existent sample {sample_data['sample_token']}")
                return
                
            if sample_data['calibrated_sensor_token'] not in calibrated_sensor_tokens:
                print(f"Error: Sample data {sample_data['token']} references non-existent calibrated sensor {sample_data['calibrated_sensor_token']}")
                return
                
        # Validate instance-annotation relationships
        for annotation in self.sample_annotations:
            if annotation['instance_token'] not in instance_tokens:
                print(f"Error: Annotation {annotation['token']} references non-existent instance {annotation['instance_token']}")
                return
                
            if annotation['sample_token'] not in sample_tokens:
                print(f"Error: Annotation {annotation['token']} references non-existent sample {annotation['sample_token']}")
                return

        # Write JSON files only to version directory
        for table_name, data in tables.items():
            self._write_output(table_name, data)
            
        print("\nAll tables written successfully")

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
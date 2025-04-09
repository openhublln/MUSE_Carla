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

class NuScenesConverter:
    """Converts CARLA sensor data to NuScenes format."""
    
    def __init__(self, config_path: str):
        """Initialize the converter with the given config file.
        
        Args:
            config_path: Path to the converter_config.yml file
        """
        self.config = self._load_config(config_path)
        self.version = self.config['version']
        self.input_base = Path(self.config['input']['base_dir'])
        self.output_base = Path(self.config['output']['base_dir'])
        self.scene_folders = self.config['input']['scene_folders']
        
        # Ensure output directory exists
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage for various records
        self._init_storage()

        # Parse configuration files
        self._parse_config_files()
    
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

        for sensor_name, files in data_files.items():
            for file_path in files:
                # Extract timestamp from the file name (assumes format: <timestamp>_suffix.ext)
                timestamp = int(Path(file_path).stem.split('_')[0])

                if timestamp not in organized_data:
                    organized_data[timestamp] = {}

                organized_data[timestamp][sensor_name] = file_path

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

        # Ensure timestamps are sorted (should be if coming from sorted dict keys)
        timestamps.sort()

        interval = int(1e6 / target_rate)  # Convert rate to microseconds interval
        keyframes = [timestamps[0]]  # Always include the first timestamp

        last_keyframe_ts = timestamps[0]
        for ts in timestamps[1:]:
            # Select if the interval has passed since the *last selected keyframe*
            if ts - last_keyframe_ts >= interval:
                keyframes.append(ts)
                last_keyframe_ts = ts # Update the last selected timestamp

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
            for timestamp in keyframes:
                sample_token = self._generate_token() # Generate token here
                sample = {
                    "token": sample_token,
                    "timestamp": timestamp,
                    "scene_token": scene_token,
                    "prev": "", # Will be filled later
                    "next": "", # Will be filled later
                }
                scene_samples.append(sample)
                # Add token mapping for easy lookup if needed later
                self.token_maps['sample'][timestamp] = sample_token

            return scene_samples

    def convert_scene(self, scene_folder: str):
        """Convert a single scene folder to NuScenes format.

        Args:
            scene_folder: Name of the scene folder to convert.
        """
        print(f"Converting scene: {scene_folder}")

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

        # Select keyframes
        target_rate = self.config['output']['keyframe_rate']
        keyframes = self._select_keyframes(list(organized_data.keys()), target_rate) # Pass only keys
        if not keyframes:
             print(f"Warning: No keyframes selected for scene {scene_folder}. Skipping sample generation.")
             # Still generate the scene entry, but with 0 samples
             scene_token = self._generate_token()
             self.token_maps['scene'][scene_folder] = scene_token # Map scene name to token
             self.scenes.append({
                 "token": scene_token,
                 "log_token": "", # Assign log token later if needed
                 "name": scene_folder,
                 "description": f"Scene data for {scene_folder} (no keyframes)",
                 "nbr_samples": 0,
                 "first_sample_token": "",
                 "last_sample_token": "",
             })
             return

        # Generate scene token and map it
        scene_token = self._generate_token()
        self.token_maps['scene'][scene_folder] = scene_token # Map scene name to token

        # Ensure all timestamps are processed
        for timestamp in sorted(organized_data.keys()):
            if timestamp not in keyframes:
                keyframes.append(timestamp)  # Add missing timestamps to keyframes

        # Generate sample entries for *this scene*
        scene_samples = self._generate_sample_entries(keyframes, scene_token)

        # --- Link prev/next within the scene's samples ---
        if scene_samples:
            print(f"Generated {len(scene_samples)} samples for scene {scene_folder}.")
            for i in range(len(scene_samples)):
                if i > 0:
                    scene_samples[i]["prev"] = scene_samples[i-1]["token"]
                else:
                    scene_samples[i]["prev"] = ""  # First sample has no prev

                if i < len(scene_samples) - 1:
                    scene_samples[i]["next"] = scene_samples[i+1]["token"]
                else:
                    scene_samples[i]["next"] = ""  # Last sample has no next

            # Append the correctly linked samples to the global list
            self.samples.extend(scene_samples)  # Ensure all samples are added to the global list

            # --- Create the scene record ---
            self.scenes.append({
                "token": scene_token,
                "log_token": "",  # Assign log token later if needed
                "name": scene_folder,
                "description": f"Scene data for {scene_folder}",
                "nbr_samples": len(scene_samples),
                "first_sample_token": scene_samples[0]["token"],
                "last_sample_token": scene_samples[-1]["token"],
            })
        else:
            # Handle case where keyframes were selected but sample generation failed (shouldn't happen with current logic)
             print(f"Warning: Keyframes selected but no samples generated for {scene_folder}. Creating scene entry with 0 samples.")
             self.scenes.append({
                 "token": scene_token,
                 "log_token": "", # Assign log token later if needed
                 "name": scene_folder,
                 "description": f"Scene data for {scene_folder} (generation failed)",
                 "nbr_samples": 0,
                 "first_sample_token": "",
                 "last_sample_token": "",
             })

    def convert_all(self):
        """Convert all scenes specified in the config."""
        for scene_folder in self.scene_folders:
            print(f"Processing scene: {scene_folder}")
            self.convert_scene(scene_folder)

        # Ensure all scenes and samples are written correctly
        print(f"DEBUG: Total scenes processed: {len(self.scenes)}")
        print(f"DEBUG: Total samples generated: {len(self.samples)}")
        self._write_all_tables()  # Write all tables to include data from all scenes

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
            'sensor': self.sensors,
            'visibility': self.visibilities,
        }

        for table_name, data in tables.items():
            self._write_output(table_name, data)

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
from typing import List, Dict, Set
from pathlib import Path
from nuscene_utils import generate_token

class SampleGenerator:
    def __init__(self, converter):
        self.converter = converter

    def select_keyframes(self, timestamps: List[int], target_rate: float = 2.0) -> List[int]:
        """Select keyframes at the target rate from the higher-frequency data.

        Args:
            timestamps: A sorted list of all unique timestamps.
            target_rate: The target keyframe rate in Hz.

        Returns:
            A list of selected keyframe timestamps.
        """
        if not timestamps:
            return []

        # Sort timestamps and remove duplicates
        timestamps = sorted(set(timestamps))
        
        # Get all timestamps that have data files
        data_timestamps = set()
        for scene_folder_name in self.converter.scene_folders:
            scene_path = self.converter.input_base / scene_folder_name
            for sensor_folder in scene_path.iterdir():
                if not sensor_folder.is_dir():
                    continue
                # Get all data files (excluding bbox files)
                data_files = [f for f in sensor_folder.glob("*.*") 
                            if not f.name.endswith(('_bbox.json', '_3dbbox.json'))]
                for data_file in data_files:
                    try:
                        ts = int(data_file.stem.split('_')[0])
                        data_timestamps.add(ts)
                    except (ValueError, IndexError):
                        continue

        # Only keep timestamps that have actual data files
        valid_timestamps = [ts for ts in timestamps if ts in data_timestamps]
        if not valid_timestamps:
            return []

        # If target rate is 20Hz (same as CARLA), keep all timestamps
        if abs(target_rate - 20.0) < 0.1:  # Using small epsilon for float comparison
            return valid_timestamps

        # For other rates, downsample
        interval = int(1000 / target_rate)
        keyframes = [valid_timestamps[0]]
        last_keyframe = valid_timestamps[0]

        for ts in valid_timestamps[1:]:
            time_diff = ts - last_keyframe
            
            if time_diff >= interval:
                candidates = [t for t in valid_timestamps if t > last_keyframe and t <= ts]
                if not candidates:
                    continue

                next_keyframe = min(candidates, key=lambda x: abs(x - (last_keyframe + interval)))
                
                if next_keyframe not in keyframes:
                    keyframes.append(next_keyframe)
                    last_keyframe = next_keyframe

        return keyframes

    def generate_sample_entries(self, keyframes: List[int], scene_token: str) -> List[dict]:
        """Generate `sample.json` entries for the selected keyframes for a single scene.

        Args:
            keyframes: A list of selected keyframe timestamps for this scene.
            scene_token: The token of the scene to which these samples belong.

        Returns:
            A list of sample dictionaries generated for this scene.
        """
        scene_samples: List[Dict] = []
        for i, timestamp in enumerate(keyframes):
            sample_token = generate_token()
            sample = {
                "token": sample_token,
                "timestamp": timestamp,
                "scene_token": scene_token,
                "prev": scene_samples[-1]["token"] if i > 0 else "",
                "next": "",
            }
            if i > 0:
                scene_samples[-1]["next"] = sample_token
            
            scene_samples.append(sample)
            self.converter.token_maps['sample'][timestamp] = sample_token

        return scene_samples 
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

        timestamps.sort()
        interval = int(1000 / target_rate)  # Convert to milliseconds
        keyframes = [timestamps[0]]
        
        bbox_timestamps: Set[int] = set()
        # Assuming self.converter.input_base is a Path object
        # Iterate through all scene folders detected by the converter
        for scene_folder_name in self.converter.scene_folders: # Iterate over detected scene folders
            scene_path = self.converter.input_base / scene_folder_name
            for sensor_folder in scene_path.iterdir(): # Iterate over sensor folders within each scene
                if not sensor_folder.is_dir():
                    continue
                # Check for 3dbbox files in each sensor folder
                bbox_files = list(sensor_folder.glob("*_3dbbox.json"))
                for bbox_file in bbox_files:
                    try:
                        if bbox_file.stat().st_size > 2:
                            ts = int(bbox_file.stem.split('_')[0])
                            bbox_timestamps.add(ts)
                    except (ValueError, IndexError):
                        continue
        
        last_keyframe = timestamps[0]
        for ts in timestamps[1:]:
            time_diff = ts - last_keyframe
            
            if time_diff >= interval:
                candidates = [t for t in timestamps if t > last_keyframe and t <= ts]
                if not candidates: continue # Should not happen if timestamps is not empty and sorted

                bbox_candidates = [t for t in candidates if t in bbox_timestamps]
                if bbox_candidates:
                    next_keyframe = min(bbox_candidates, key=lambda x: abs(x - (last_keyframe + interval)))
                else:
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
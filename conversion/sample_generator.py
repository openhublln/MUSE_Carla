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
        
        # Use provided timestamps from the current scene; they already reflect available data
        valid_timestamps = timestamps
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
                "timestamp": self.converter.epoch_base_us + timestamp * 1000,
                "scene_token": scene_token,
                "prev": scene_samples[-1]["token"] if i > 0 else "",
                "next": "",
            }
            if i > 0:
                scene_samples[-1]["next"] = sample_token
            
            scene_samples.append(sample)
            self.converter.token_maps['sample'][timestamp] = sample_token

        return scene_samples 
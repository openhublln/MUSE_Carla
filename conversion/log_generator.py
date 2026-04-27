import json
from .nuscene_utils import generate_token

class LogGenerator:
    def __init__(self, converter):
        self.converter = converter

    def generate_log_entry(self):
        """Generate the NuScenes log entry from log_info.json."""
        # Try to determine location from map files
        map_location = self._get_map_location()
        
        log_info_path = self.converter.input_base / "log_info.json"
        if not log_info_path.exists():
            print(f"Warning: log_info.json not found at {log_info_path}. Using auto-detected location: {map_location}")
            log_info = {}
        else:
            with open(log_info_path, 'r') as f:
                log_info = json.load(f)
        
        # Use detected map location if not specified in log_info
        location = log_info.get("location", map_location)
        
        log_entry = {
            "token": generate_token(),
            "logfile": log_info.get("logfile", "carla_simulation_log"),
            "vehicle": log_info.get("vehicle", "carla_ego_vehicle"),
            "date_captured": log_info.get("date_captured", "2024-01-01"),
            "location": location
        }
        self.converter.logs = [log_entry]
        self.converter.log_token = log_entry["token"]
        print(f"Log entry created with location: {location}")

    def _get_map_location(self):
        """Detect map location from generated map files."""
        try:
            source_map_dir = self.converter.input_base
            source_map_json_files = [f for f in source_map_dir.glob('*.json') if f.stem not in ['log_info', 'config']]
            
            if source_map_json_files:
                source_json_path = source_map_json_files[0]
                with open(source_json_path, 'r') as f:
                    map_meta = json.load(f)
                # Return the original CARLA map name to preserve map identity
                original_map_name = map_meta.get('original_carla_map', source_json_path.stem)
                return original_map_name
        except Exception as e:
            print(f"Warning: Could not detect map location: {e}")
        
        return "carla-town"  # Default fallback

    def assign_log_token_to_scenes(self):
        """Assign the log token to all scene records."""
        # Ensure log_token exists, even if log_info.json was missing
        log_token_to_assign = getattr(self.converter, "log_token", "")
        for scene in self.converter.scenes:
            scene["log_token"] = log_token_to_assign 
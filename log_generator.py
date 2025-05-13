import json
from nuscene_utils import generate_token

class LogGenerator:
    def __init__(self, converter):
        self.converter = converter

    def generate_log_entry(self):
        """Generate the NuScenes log entry from log_info.json."""
        log_info_path = self.converter.input_base / "log_info.json"
        if not log_info_path.exists():
            print(f"Warning: log_info.json not found at {log_info_path}. Skipping log.json generation.")
            self.converter.log_token = "" # Ensure log_token is initialized
            self.converter.logs = []
            return
        with open(log_info_path, 'r') as f:
            log_info = json.load(f)
        log_entry = {
            "token": generate_token(),
            "logfile": log_info.get("logfile", ""),
            "vehicle": log_info.get("vehicle", ""),
            "date_captured": log_info.get("date_captured", ""),
            "location": log_info.get("location", "")
        }
        self.converter.logs = [log_entry]
        self.converter.log_token = log_entry["token"]

    def assign_log_token_to_scenes(self):
        """Assign the log token to all scene records."""
        # Ensure log_token exists, even if log_info.json was missing
        log_token_to_assign = getattr(self.converter, "log_token", "")
        for scene in self.converter.scenes:
            scene["log_token"] = log_token_to_assign 
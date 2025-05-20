import json
from typing import List, Dict
from nuscene_utils import generate_token, euler_to_quaternion

class SensorCalibratedGenerators:
    def __init__(self, converter):
        self.converter = converter

    def generate_sensor_entries(self):
        sensor_mappings = self.converter.config.get("sensor_mappings", {})
        simulation_sensors = self.converter.sim_config.get("sensors", [])
        self.converter.sensors = []
        # Map sensor name to token for robust linking
        self.converter.sensor_name_to_token = {}
        for sensor in simulation_sensors:
            sensor_name = sensor["name"]
            sensor_type = sensor["type"]
            if sensor_type in sensor_mappings and sensor_name in sensor_mappings[sensor_type]:
                channel = sensor_mappings[sensor_type][sensor_name]
                token = generate_token()
                sensor_entry = {
                    "token": token,
                    "channel": channel,
                    "modality": sensor_type
                }
                self.converter.sensors.append(sensor_entry)
                self.converter.sensor_name_to_token[sensor_name] = token
        sensor_output_path = self.converter.output_base / self.converter.version / 'sensor.json'
        # Ensure the version directory exists
        sensor_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sensor_output_path, 'w') as f:
            json.dump(self.converter.sensors, f, indent=2)

    def generate_calibrated_sensors(self):
        self.converter.calibrated_sensors = []
        simulation_sensors = self.converter.sim_config.get("sensors", [])
        # Build sensor name to token mapping from sensor.json
        sensor_json_path = self.converter.output_base / self.converter.version / 'sensor.json'
        with open(sensor_json_path, 'r') as f:
            sensor_json = json.load(f)
        sensor_name_to_token = {entry["channel"]: entry["token"] for entry in sensor_json}
        sensor_mappings = self.converter.config.get("sensor_mappings", {})
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

                rotation_quaternion = euler_to_quaternion(roll_nusc, pitch_nusc, yaw_nusc)

                calibrated_sensor_entry = {
                    "token": generate_token(),
                    "sensor_token": sensor_token,
                    "translation": translation,
                    "rotation": rotation_quaternion,
                    "camera_intrinsic": []
                }
                self.converter.calibrated_sensors.append(calibrated_sensor_entry)
        # Write calibrated_sensor.json immediately after generating
        calibrated_output_path = self.converter.output_base / self.converter.version / 'calibrated_sensor.json'
        # Ensure the version directory exists
        calibrated_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(calibrated_output_path, 'w') as f:
            json.dump(self.converter.calibrated_sensors, f, indent=2) 
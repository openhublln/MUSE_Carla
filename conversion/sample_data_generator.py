import json
from pathlib import Path
from bisect import bisect_left
from .nuscene_utils import generate_token, carla_rotation_to_nuscenes_quaternion, adjust_z_for_ego_pose
from PIL import Image
import shutil
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

class SampleDataGenerator:
    def __init__(self, converter):
        self.converter = converter
        # Cache for sensor and calibrated sensor tables to avoid repeated disk reads
        self._sensor_json_cache = None
        self._calibrated_json_cache = None
        # Configurable worker count via converter_config.yml → performance.max_workers
        perf_cfg = self.converter.config.get("performance", {}) if hasattr(self.converter, "config") else {}
        default_workers = max(1, min(32, (os.cpu_count() or 4) * 2))
        try:
            self.max_workers = int(perf_cfg.get("max_workers", default_workers))
            if self.max_workers < 1:
                self.max_workers = 1
        except Exception:
            self.max_workers = default_workers
        # Verbose gate to throttle per-file prints
        self.verbose = bool(perf_cfg.get("verbose", False))

    def ensure_ego_poses_for_scene_sample_data(self, scene_folder: str):
        """Guarantee every sample_data timestamp in this scene has an ego_pose entry.
        
        After generate_sample_data_entries() runs, any sample_data whose timestamp is
        absent from ego_pose/*.json (off-by-one tick races, rounding, etc.) would
        produce an empty ego_pose_token and fail NuScenes validation.
        
        Strategy: collect all sample_data timestamps for this scene, find those
        missing an ego_pose, then interpolate (or duplicate nearest) from existing
        ego_poses recorded for the same scene.
        """
        # Gather existing ego_pose entries for this scene
        scene_ep_map = {}  # timestamp -> ego_pose entry
        for ep in self.converter.ego_poses:
            key = (scene_folder, ep['timestamp'])
            if key in self.converter.token_maps.get('ego_pose', {}):
                scene_ep_map[ep['timestamp']] = ep

        if not scene_ep_map:
            return  # Nothing to interpolate from

        sorted_ep_ts = sorted(scene_ep_map.keys())

        # Collect all sample_data timestamps for this scene (via sample_token linkage)
        scene_sample_tokens = {s['token'] for s in self.converter.samples
                               if s.get('scene_token') == self.converter.token_maps['scene'].get(scene_folder)}

        missing_timestamps = set()
        for sd in self.converter.sample_data:
            if sd.get('sample_token') not in scene_sample_tokens:
                continue
            ts = sd['timestamp']
            if (scene_folder, ts) not in self.converter.token_maps.get('ego_pose', {}):
                missing_timestamps.add(ts)

        for ts in missing_timestamps:
            # Find nearest existing ego_pose timestamp
            idx = bisect_left(sorted_ep_ts, ts)
            if idx == 0:
                nearest_ts = sorted_ep_ts[0]
            elif idx >= len(sorted_ep_ts):
                nearest_ts = sorted_ep_ts[-1]
            else:
                before, after = sorted_ep_ts[idx - 1], sorted_ep_ts[idx]
                nearest_ts = before if (ts - before) <= (after - ts) else after

            nearest_ep = scene_ep_map[nearest_ts]
            token = generate_token()
            new_ep = {
                "token": token,
                "timestamp": ts,
                "translation": list(nearest_ep['translation']),
                "rotation": list(nearest_ep['rotation']),
            }
            self.converter.ego_poses.append(new_ep)
            self.converter.token_maps['ego_pose'][(scene_folder, ts)] = token
            scene_ep_map[ts] = new_ep

        # Patch empty ego_pose_token in sample_data entries for this scene
        ep_token_map = self.converter.token_maps.get('ego_pose', {})
        for sd in self.converter.sample_data:
            if sd.get('sample_token') not in scene_sample_tokens:
                continue
            if not sd.get('ego_pose_token'):
                sd['ego_pose_token'] = ep_token_map.get((scene_folder, sd['timestamp']), "")

    def generate_ego_poses_for_scene(self, scene_folder: str):
        """Generate ego poses for a scene. This is called before annotation generation."""
        scene_path = self.converter.input_base / scene_folder
        ego_pose_dir = scene_path / self.converter.EGO_POSE_FOLDER
        
        if ego_pose_dir.exists():
            for pose_file in ego_pose_dir.glob("*.json"):
                try:
                    timestamp = int(pose_file.stem)
                    with open(pose_file, 'r') as f:
                        pose_data = json.load(f)
                        
                    # Convert CARLA coordinates to NuScenes coordinate system
                    # CARLA: X-forward, Y-right, Z-up
                    # NuScenes: X-forward, Y-left, Z-up (negate Y)
                    translation = [
                        float(pose_data["translation"]["x"]),
                        -float(pose_data["translation"]["y"]),  # Negate Y for coordinate system conversion
                        float(pose_data["translation"]["z"])
                    ]
                    
                    # Convert CARLA rotation to NuScenes using proper coordinate transformation
                    roll = float(pose_data["rotation"]["roll"])
                    pitch = float(pose_data["rotation"]["pitch"]) 
                    yaw = float(pose_data["rotation"]["yaw"])
                    
                    rotation = carla_rotation_to_nuscenes_quaternion(roll, pitch, yaw)
                    
                    # Use actual world coordinates for ego pose (not hardcoded origin)
                    token = generate_token()
                    ego_pose_entry = {
                        "token": token,
                        "timestamp": timestamp,
                        "translation": translation,  # Use actual world position
                        "rotation": rotation  # Use actual world orientation
                    }
                    
                    self.converter.ego_poses.append(ego_pose_entry)
                    # Track per-scene ego pose token to avoid cross-scene collisions
                    try:
                        self.converter.token_maps['ego_pose'][(scene_folder, timestamp)] = token
                    except Exception:
                        pass
                    
                except Exception as e:
                    print(f"Warning: Error processing ego pose file {pose_file}: {e}")
                    continue

    def generate_sample_data_entries(self, scene_folder: str, scene_token: str):
        scene_path = self.converter.input_base / scene_folder
        sensor_json_path = self.converter.output_base / self.converter.version / 'sensor.json'
        calibrated_json_path = self.converter.output_base / self.converter.version / 'calibrated_sensor.json'
        
        try:
            if self._sensor_json_cache is None:
                with open(sensor_json_path, 'r') as f:
                    self._sensor_json_cache = json.load(f)
            if self._calibrated_json_cache is None:
                with open(calibrated_json_path, 'r') as f:
                    self._calibrated_json_cache = json.load(f)
            sensor_json = self._sensor_json_cache
            calibrated_json = self._calibrated_json_cache
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
        sorted_keyframes = sorted(keyframe_timestamps)
        def nearest_keyframe(ts: int):
            if not sorted_keyframes:
                return None
            i = bisect_left(sorted_keyframes, ts)
            if i == 0:
                return sorted_keyframes[0]
            if i == len(sorted_keyframes):
                return sorted_keyframes[-1]
            before, after = sorted_keyframes[i-1], sorted_keyframes[i]
            return before if (ts - before) <= (after - ts) else after
        
        all_sensor_timestamps = set()
        for sensor in simulation_sensors:
            sensor_name = sensor["name"]
            sensor_data_folder = scene_path / sensor_name
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
            files = [f for f in sensor_data_folder.glob(f"*{ext}") if f.is_file()]
            for file in files:
                try:
                    timestamp = int(file.stem)
                    all_sensor_timestamps.add(timestamp)
                except ValueError:
                    # Attempt to parse if filename is <timestamp>_suffix.ext
                    try:
                        timestamp = int(file.stem.split('_')[0])
                        all_sensor_timestamps.add(timestamp)
                    except (ValueError, IndexError):
                        continue

        ego_pose_dir = scene_path / self.converter.EGO_POSE_FOLDER
        ego_pose_timestamps = {}
        if ego_pose_dir.exists():
            # Map only timestamps present in this scene's ego_pose folder
            for pose_file in ego_pose_dir.glob("*.json"):
                try:
                    ts = int(pose_file.stem)
                    token = self.converter.token_maps.get('ego_pose', {}).get((scene_folder, ts), "")
                    if token:
                        ego_pose_timestamps[ts] = token
                except Exception:
                    continue

        for sensor in simulation_sensors:
            sensor_name = sensor["name"]
            sensor_type = sensor["type"]
            channel = sensor_folder_to_channel.get(sensor_name)
            if not channel:
                continue
            
            sensor_token_val = channel_to_sensor_token.get(channel, "")
            calibrated_sensor_token = sensor_token_to_calibrated_token.get(sensor_token_val, "")
            current_sensor_folder = scene_path / sensor_name

            if not current_sensor_folder.exists() or not current_sensor_folder.is_dir():
                continue

            fileformat, width, height, ext = "", None, None, ""
            if sensor_type == "camera":
                ext = ".png"
                fileformat = "jpg"  # Changed from png to jpg for nuScenes compatibility
                width = int(float(sensor.get("attributes", {}).get("image_size_x", 0)))
                height = int(float(sensor.get("attributes", {}).get("image_size_y", 0)))
            elif sensor_type == "lidar":
                ext = ".npy"
                fileformat = "pcd.bin"  # NuScenes uses .pcd.bin for LIDAR
            elif sensor_type == "radar":
                ext = ".npy"
                fileformat = "pcd"
            elif sensor_type == "semantic_lidar":
                ext = ".npy"
                fileformat = "pcd.bin"
            elif sensor_type in ("imu", "gnss"):
                ext = ".json"
                fileformat = "json"
            else:
                continue
            
            files = sorted([f for f in current_sensor_folder.glob(f"*{ext}") if f.is_file()])
            if not files:
                continue

            def _process_one(file_path: Path):
                try:
                    # Parse timestamp
                    try:
                        timestamp_local = int(file_path.stem)
                    except ValueError:
                        try:
                            timestamp_local = int(file_path.stem.split('_')[0])
                        except (ValueError, IndexError):
                            return None

                    closest_keyframe_ts = nearest_keyframe(timestamp_local)
                    if closest_keyframe_ts is None:
                        return None
                    sample_token_local = sample_timestamps[closest_keyframe_ts]

                    # Determine keyframe status and routing folder (samples vs sweeps)
                    is_key_frame_local = timestamp_local in keyframe_timestamps
                    routing = "samples" if is_key_frame_local else "sweeps"

                    # NuScenes filename format: <logfile>__<CHANNEL>__<timestamp_us>.<ext>
                    # timestamps are stored in ms → convert to µs (* 1000)
                    logfile = self.converter.logs[0].get("logfile", "") if self.converter.logs else ""
                    timestamp_us = timestamp_local * 1000

                    if sensor_type == "camera":
                        out_filename = f"{logfile}__{channel}__{timestamp_us}.jpg"
                    elif sensor_type == "lidar":
                        out_filename = f"{logfile}__{channel}__{timestamp_us}.pcd.bin"
                    elif sensor_type == "radar":
                        out_filename = f"{logfile}__{channel}__{timestamp_us}.pcd"
                    elif sensor_type == "semantic_lidar":
                        out_filename = f"{logfile}__{channel}__{timestamp_us}.pcd.bin"
                    else:
                        out_filename = f"{logfile}__{channel}__{timestamp_us}{ext}"

                    filename_local = f"{routing}/{channel}/{out_filename}"
                    target_dir_local = self.converter.output_base / routing / channel
                    target_dir_local.mkdir(parents=True, exist_ok=True)
                    target_file_local = target_dir_local / out_filename

                    ego_pose_token_local = ego_pose_timestamps.get(timestamp_local, "")

                    # Convert/copy
                    if sensor_type == "camera":
                        img = Image.open(file_path)
                        img.convert('RGB').save(target_file_local, 'JPEG')
                        if self.verbose:
                            print(f"Converted and copied {file_path.name} to {target_file_local.name}")
                    elif sensor_type == "lidar":
                        points = np.load(file_path)
                        try:
                            if points.shape[1] >= 2:
                                points[:, 1] *= -1
                        except Exception:
                            pass
                        if points.ndim == 1:
                            points = points.reshape(-1, 3)
                        if points.shape[1] < 5:
                            padded_points = np.zeros((points.shape[0], 5), dtype=points.dtype)
                            padded_points[:, :points.shape[1]] = points
                            points = padded_points
                        points.astype(np.float32).tofile(target_file_local)
                        if self.verbose:
                            print(f"Converted {file_path.name} to {target_file_local.name}")
                    elif sensor_type == "radar":
                        points = np.load(file_path)
                        if points.shape[1] < 18:
                            padded_points = np.zeros((points.shape[0], 18))
                            padded_points[:, :points.shape[1]] = points
                            points = padded_points
                        with open(target_file_local, 'wb') as f:
                            f.write(b"# .PCD v0.7 - Point Cloud Data file format\n")
                            f.write(b"VERSION 0.7\n")
                            f.write(b"FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms\n")
                            f.write(b"SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1\n")
                            f.write(b"TYPE F F F I I F F F F F I I I I I I I I\n")
                            f.write(b"COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n")
                            f.write(f"WIDTH {points.shape[0]}\n".encode())
                            f.write(b"HEIGHT 1\n")
                            f.write(b"VIEWPOINT 0 0 0 1 0 0 0\n")
                            f.write(f"POINTS {points.shape[0]}\n".encode())
                            f.write(b"DATA binary\n")
                            points.astype(np.float32).tofile(f)
                        if self.verbose:
                            print(f"Converted {file_path.name} to {target_file_local.name}")
                    elif sensor_type == "semantic_lidar":
                        points = np.load(file_path)
                        points.astype(np.float32).tofile(target_file_local)
                        if self.verbose:
                            print(f"Converted {file_path.name} to {target_file_local.name}")
                    else:
                        shutil.copy2(file_path, target_file_local)
                        if self.verbose:
                            print(f"Copied {file_path.name} to {target_file_local.name}")

                    token_local = generate_token()
                    entry_local = {
                        "token": token_local,
                        "sample_token": sample_token_local,
                        "ego_pose_token": ego_pose_token_local,
                        "calibrated_sensor_token": calibrated_sensor_token,
                        "filename": filename_local,
                        "fileformat": fileformat,
                        "width": width,
                        "height": height,
                        "timestamp": timestamp_local,
                        "is_key_frame": is_key_frame_local,
                        "next": "",
                        "prev": ""
                    }
                    return (timestamp_local, entry_local)
                except Exception as e:
                    print(f"Warning: Error processing file {file_path} for sensor {sensor_name}: {e}")
                    return None

            max_workers = self.max_workers
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(_process_one, f): f for f in files}
                for future in as_completed(future_map):
                    res = future.result()
                    if res is not None:
                        results.append(res)

            # Preserve chronological linking
            results.sort(key=lambda x: x[0])
            ordered_entries = [e for _, e in results]
            for i, entry in enumerate(ordered_entries):
                if i > 0:
                    entry["prev"] = ordered_entries[i-1]["token"]
                if i < len(ordered_entries) - 1:
                    entry["next"] = ordered_entries[i+1]["token"]
                self.converter.sample_data.append(entry)
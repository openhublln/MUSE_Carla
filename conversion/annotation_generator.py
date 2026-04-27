import json
import numpy as np
from typing import List, Dict
from scipy.spatial import cKDTree
# Assuming nuscene_utils and carla types are available or passed through self.converter
from nuscene_utils import generate_token, carla_rotation_to_nuscenes_quaternion, carla_camera_rotation_to_nuscenes_quaternion, count_points_in_box, transform_points_to_global, transform_radar_points_to_global, transform_box_to_ego_frame
import carla # For carla.Transform type hint if needed directly

class AnnotationGenerator:
    def __init__(self, converter):
        self.converter = converter
        # OPTION: Set to True to test camera-specific coordinate transformations for annotations
        # This is an alternative fix if the primary fix in sensor_calibrated_generators.py doesn't work
        self.use_camera_specific_coordinates = False  # BACK TO BASICS: Use general coordinates

    def _infer_vehicle_state(self, velocity_magnitude: float, is_static: bool = False) -> str:
        """Return a NuScenes vehicle attribute name.

        Static world vehicles (pre-placed map props with is_static=True) are always
        'vehicle.parked' regardless of velocity (which is always 0 for them anyway).
        Dynamic vehicles use a simple velocity threshold.
        """
        if is_static:
            return "vehicle.parked"
        MOVING_THRESHOLD = 0.5
        if velocity_magnitude > MOVING_THRESHOLD:
            return "vehicle.moving"
        else:
            return "vehicle.stopped"

    def _get_visibility_level(self, visibility: float) -> str:
        """Return level string matching token_maps['visibility'] keys (v0-40 … v80-100)."""
        if visibility < 40:
            return "v0-40"
        elif visibility < 60:
            return "v40-60"
        elif visibility < 80:
            return "v60-80"
        else:
            return "v80-100"

    def _count_total_cameras(self, scene_folder: str) -> int:
        scene_path = self.converter.input_base / scene_folder
        camera_count = 0
        for sensor_config in self.converter.sim_config.get("sensors", []):
            if sensor_config.get("type") == "camera":
                if (scene_path / sensor_config["name"]).is_dir():
                    camera_count += 1
        return camera_count

    def _compute_average_visibility(self, scene_folder: str, timestamp: int, actor_id: int) -> float:
        # Use converter-level cache to avoid recomputation
        cache_key = (scene_folder, timestamp, actor_id)
        cached = self.converter._visibility_cache.get(cache_key)
        if cached is not None:
            return cached
        scene_path = self.converter.input_base / scene_folder
        total_visibility = 0.0
        total_cameras = self._count_total_cameras(scene_folder)
        if total_cameras == 0:
            self.converter._visibility_cache[cache_key] = 0.0
            return 0.0
        
        for sensor_config in self.converter.sim_config.get("sensors", []):
            if sensor_config.get("type") == "camera":
                sensor_name = sensor_config["name"]
                # Ensure this sensor folder exists for the current scene
                sensor_data_folder = scene_path / sensor_name
                if not sensor_data_folder.is_dir():
                    continue

                bbox_file = sensor_data_folder / f"{timestamp}_3dbbox.json"
                if not bbox_file.exists():
                    continue
                
                try:
                    with open(bbox_file, 'r') as f:
                        bbox_data = json.load(f)
                    visibility_for_actor = 0.0
                    for annotation in bbox_data:
                        if annotation.get("actor_id") == actor_id and "visibility" in annotation:
                            visibility_for_actor = annotation["visibility"]
                            break
                    total_visibility += visibility_for_actor
                except Exception as e:
                    # print(f"Warning: Error reading bbox file {bbox_file} for visibility: {e}")
                    continue
        avg = total_visibility / total_cameras if total_cameras > 0 else 0.0
        self.converter._visibility_cache[cache_key] = avg
        return avg

    def generate_sample_annotations(self, scene_folder: str, scene_token: str):
        sample_token_map = {entry["timestamp"]: entry["token"] 
                            for entry in self.converter.samples if entry["scene_token"] == scene_token}
        
        # print(f"[Debug SA {scene_folder}]: Found {len(sample_token_map)} samples for this scene.") # DEBUG
        if not sample_token_map:
            # print(f"[Debug SA {scene_folder}]: No samples in sample_token_map, exiting annotation generation for this scene.") # DEBUG
            return

        instance_annotations = {} 
        # Per-scene caches to avoid reloading arrays repeatedly
        lidar_points_cache = {}
        lidar_kd_cache = {}
        radar_points_cache = {}
        radar_kd_cache = {}
        scene_path = self.converter.input_base / scene_folder
        # found_any_bbox_files = False # DEBUG
        # annotations_created_for_scene = 0 # DEBUG
        
        for sensor_dir in scene_path.iterdir():
            if not sensor_dir.is_dir():
                continue
            
            bbox_files = list(sensor_dir.glob("*_3dbbox.json"))
            # if bbox_files:
                # found_any_bbox_files = True # DEBUG
                # print(f"[Debug SA {scene_folder}]: Processing sensor dir: {sensor_dir.name}, found {len(bbox_files)} bbox files.") # DEBUG
            
            for bbox_file in bbox_files:
                try:
                    if bbox_file.stat().st_size <= 2:
                        # print(f"[Debug SA {scene_folder}]: Skipping empty bbox file: {bbox_file}") # DEBUG
                        continue
                    timestamp = int(bbox_file.stem.split('_')[0])
                    sample_token = sample_token_map.get(timestamp)
                    if not sample_token:
                        # print(f"[Debug SA {scene_folder}]: No sample_token for timestamp {timestamp} from file {bbox_file.name}") # DEBUG (can be verbose)
                        continue
                        
                    with open(bbox_file, 'r') as f:
                        bbox_data_list = json.load(f)
                    
                    if not bbox_data_list:
                        # print(f"[Debug SA {scene_folder}]: bbox_data_list is empty for {bbox_file.name}") # DEBUG
                        continue
                        
                    for annotation_data in bbox_data_list:
                        actor_id = annotation_data.get("actor_id")
                        if actor_id is None: 
                            # print(f"[Debug SA {scene_folder}]: actor_id is None in {bbox_file.name}") # DEBUG (can be verbose)
                            continue
                            
                        instance_token = self.converter.token_maps['instance'].get((scene_folder, actor_id))
                        if not instance_token: 
                            # print(f"[Debug SA {scene_folder}]: No instance_token for actor_id {actor_id} (scene: {scene_folder}) in {bbox_file.name}") # DEBUG
                            # print(f"[Debug SA {scene_folder}]: Current instance map: {self.converter.token_maps['instance']}") # DEBUG (VERY verbose)
                            continue
                            
                        pose = annotation_data.get("pose", {})
                        translation_data = pose.get("translation", {})
                        rotation_data = pose.get("rotation", {})
                        
                        # Apply CARLA to NuScenes coordinate system conversion
                        # CARLA: X-forward, Y-right, Z-up → NuScenes: X-forward, Y-left, Z-up
                        box_center = [
                            float(translation_data.get("x", 0)),
                            -float(translation_data.get("y", 0)),  # Negate Y for coordinate conversion
                            float(translation_data.get("z", 0))
                        ]
                        
                        # Convert CARLA rotation to NuScenes quaternion
                        # Keep original approach that worked for LiDAR visualization
                        raw_roll = float(rotation_data.get("roll", 0))
                        raw_pitch = float(rotation_data.get("pitch", 0))
                        raw_yaw = float(rotation_data.get("yaw", 0))
                        
                        # Use standard rotation conversion (global coordinates for annotations)
                        quaternion = carla_rotation_to_nuscenes_quaternion(raw_roll, raw_pitch, raw_yaw)
                        
                        velocity = annotation_data.get("velocity", {})
                        velocity_magnitude = velocity.get("magnitude", 0.0)
                        ann_category = annotation_data.get("category", "")
                        is_static = annotation_data.get("is_static", False)
                        if ann_category.startswith("human.pedestrian"):
                            # Pedestrian attribute: moving vs standing
                            ped_state = "pedestrian.moving" if velocity_magnitude > 0.5 else "pedestrian.standing"
                            attribute_token = self.converter.attribute_name_to_token.get(ped_state, "")
                        else:
                            # Vehicle attribute: parked (static world props) / moving / stopped
                            vehicle_state = self._infer_vehicle_state(velocity_magnitude, is_static=is_static)
                            attribute_token = self.converter.attribute_name_to_token.get(vehicle_state, "")
                        
                        # Apply CARLA to NuScenes coordinate system conversion
                        # CARLA: X-forward, Y-right, Z-up
                        # NuScenes: X-forward, Y-left, Z-up (negate Y)
                        box_size = annotation_data.get("size", [0,0,0])

                        # NOTE: Quaternion already calculated above with yaw correction applied
                        # No need to recalculate - this was overwriting the corrected quaternion!
                        
                        num_lidar_pts = 0
                        # Count LIDAR points in the annotation box (both in same global coordinate system)
                        lidar_sensor_name = next((s["name"] for s in self.converter.sim_config.get("sensors", []) if s["type"] == "lidar"), "Lidar")
                        lidar_file = scene_path / lidar_sensor_name / f"{timestamp}.npy"
                        if lidar_file.exists():
                            try:
                                # Load raw LIDAR points once and invert Y once per timestamp
                                lidar_points_nuscenes = lidar_points_cache.get(timestamp)
                                if lidar_points_nuscenes is None:
                                    lidar_points_carla = np.load(lidar_file)
                                    lidar_points_nuscenes = lidar_points_carla
                                    try:
                                        lidar_points_nuscenes = lidar_points_nuscenes.astype(np.float32, copy=False)
                                    except Exception:
                                        pass
                                    lidar_points_nuscenes = lidar_points_nuscenes.copy()
                                    lidar_points_nuscenes[:, 1] *= -1  # Negate Y coordinate
                                    lidar_points_cache[timestamp] = lidar_points_nuscenes
                                # KDTree candidate filter
                                tree = lidar_kd_cache.get(timestamp)
                                if tree is None:
                                    tree = cKDTree(lidar_points_nuscenes[:, :3])
                                    lidar_kd_cache[timestamp] = tree
                                # Conservative AABB in global frame: use box center +/- size to get candidates
                                half = np.array(box_size, dtype=np.float32) / 2.0
                                aabb_min = np.array(box_center, dtype=np.float32) - half
                                aabb_max = np.array(box_center, dtype=np.float32) + half
                                # Query candidates within sphere of radius=norm(half) around center, then refine
                                radius = float(np.linalg.norm(half))
                                idxs = tree.query_ball_point(box_center, r=radius)
                                if idxs:
                                    candidates = lidar_points_nuscenes[idxs, :3]
                                    # Exact oriented box check
                                    num_lidar_pts = count_points_in_box(candidates, box_center, box_size, quaternion)
                            except Exception as e:
                                print(f"Warning: Error processing lidar points for annotation: {e}")
                        
                        num_radar_pts = 0
                        radar_sensor_configs = [s for s in self.converter.sim_config.get("sensors", []) if s["type"] == "radar"]
                        for radar_cfg in radar_sensor_configs:
                            radar_name = radar_cfg["name"]
                            radar_file = scene_path / radar_name / f"{timestamp}.npy"
                            if radar_file.exists():
                                try:
                                    key = (radar_name, timestamp)
                                    radar_points_nuscenes = radar_points_cache.get(key)
                                    if radar_points_nuscenes is None:
                                        radar_points = np.load(radar_file)
                                        sensor_transform_rel, ego_transform_abs = self.converter._get_sensor_transform(scene_folder, radar_name, timestamp)
                                        if sensor_transform_rel and ego_transform_abs:
                                            global_points_carla = transform_radar_points_to_global(radar_points, sensor_transform_rel, ego_transform_abs)
                                            if len(global_points_carla) > 0:
                                                radar_points_nuscenes = global_points_carla.astype(np.float32, copy=False)
                                                radar_points_nuscenes = radar_points_nuscenes.copy()
                                                radar_points_nuscenes[:, 1] *= -1
                                                radar_points_cache[key] = radar_points_nuscenes
                                    # KDTree candidate filter
                                    if radar_points_nuscenes is not None and len(radar_points_nuscenes) > 0:
                                        tree = radar_kd_cache.get(key)
                                        if tree is None:
                                            tree = cKDTree(radar_points_nuscenes[:, :3])
                                            radar_kd_cache[key] = tree
                                        half = np.array(box_size, dtype=np.float32) / 2.0
                                        radius = float(np.linalg.norm(half))
                                        idxs = tree.query_ball_point(box_center, r=radius)
                                        if idxs:
                                            candidates = radar_points_nuscenes[idxs, :3]
                                            num_radar_pts += count_points_in_box(candidates, box_center, box_size, quaternion)
                                except Exception as e:
                                    print(f"Warning: Error processing radar points for {radar_name} for annotation: {e}")

                        annotation_token = generate_token()
                        annotation_entry = {
                            "token": annotation_token,
                            "sample_token": sample_token,
                            "instance_token": instance_token,
                            "attribute_tokens": [attribute_token] if attribute_token else [],
                            "visibility_token": "", 
                            "translation": box_center,
                            "size": box_size,
                            "rotation": quaternion,
                            "num_lidar_pts": num_lidar_pts,
                            "num_radar_pts": num_radar_pts,
                            "next": "",
                            "prev": ""
                        }
                        
                        if instance_token not in instance_annotations:
                            instance_annotations[instance_token] = []
                        instance_annotations[instance_token].append((timestamp, annotation_token))
                        self.converter.sample_annotations.append(annotation_entry)
                        # annotations_created_for_scene +=1 # DEBUG
                except Exception as e:
                    print(f"Error processing bbox file {bbox_file} for annotations: {e}")
                    continue
        
        # if not found_any_bbox_files:
            # print(f"[Debug SA {scene_folder}]: No 3dbbox files found in any sensor directory.") # DEBUG
        # print(f"[Debug SA {scene_folder}]: Created {annotations_created_for_scene} annotation entries for this scene.") # DEBUG

        for instance_token, annotations_ts_list in instance_annotations.items():
            annotations_ts_list.sort(key=lambda x: x[0])
            for i, (_, ann_tok) in enumerate(annotations_ts_list):
                # Find the annotation in self.converter.sample_annotations to update
                annotation_to_update = next((a for a in self.converter.sample_annotations if a["token"] == ann_tok), None)
                if annotation_to_update:
                    if i > 0:
                        annotation_to_update["prev"] = annotations_ts_list[i-1][1]
                    if i < len(annotations_ts_list) - 1:
                        annotation_to_update["next"] = annotations_ts_list[i+1][1]

    def update_sample_annotations_with_visibility(self, scene_folder: str):
        scene_token = self.converter.token_maps['scene'].get(scene_folder)
        if not scene_token: return

        for annotation in self.converter.sample_annotations:
            # Ensure this annotation belongs to a sample in the current scene
            sample = next((s for s in self.converter.samples if s["token"] == annotation["sample_token"] and s["scene_token"] == scene_token), None)
            if not sample: continue

            instance_token = annotation["instance_token"]
            actor_id = next((aid for (sf, aid), tok in self.converter.token_maps['instance'].items() 
                             if tok == instance_token and sf == scene_folder), None)
            if actor_id is None: continue
            
            avg_visibility = self._compute_average_visibility(scene_folder, sample["timestamp"], actor_id)
            visibility_level = self._get_visibility_level(avg_visibility)
            visibility_token = self.converter.token_maps['visibility'].get(visibility_level)
            
            if visibility_token:
                annotation["visibility_token"] = visibility_token 
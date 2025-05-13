import json
import numpy as np
from typing import List, Dict
# Assuming nuscene_utils and carla types are available or passed through self.converter
from nuscene_utils import generate_token, euler_to_quaternion, count_points_in_box, transform_points_to_global, transform_radar_points_to_global
import carla # For carla.Transform type hint if needed directly

class AnnotationGenerator:
    def __init__(self, converter):
        self.converter = converter

    def _infer_vehicle_state(self, velocity_magnitude: float) -> str:
        MOVING_THRESHOLD = 0.5
        if velocity_magnitude > MOVING_THRESHOLD:
            return "vehicle.moving"
        else:
            return "vehicle.stopped"

    def _get_visibility_level(self, visibility: float) -> str:
        if visibility < 40:
            return "v0"
        elif visibility < 60:
            return "v1"
        elif visibility < 80:
            return "v2"
        else:
            return "v3"

    def _count_total_cameras(self, scene_folder: str) -> int:
        scene_path = self.converter.input_base / scene_folder
        camera_count = 0
        for sensor_config in self.converter.sim_config.get("sensors", []):
            if sensor_config.get("type") == "camera":
                if (scene_path / sensor_config["name"]).is_dir():
                    camera_count += 1
        return camera_count

    def _compute_average_visibility(self, scene_folder: str, timestamp: int, actor_id: int) -> float:
        scene_path = self.converter.input_base / scene_folder
        total_visibility = 0.0
        total_cameras = self._count_total_cameras(scene_folder)
        if total_cameras == 0:
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
        
        return total_visibility / total_cameras if total_cameras > 0 else 0.0

    def generate_sample_annotations(self, scene_folder: str, scene_token: str):
        sample_token_map = {entry["timestamp"]: entry["token"] 
                            for entry in self.converter.samples if entry["scene_token"] == scene_token}
        
        # print(f"[Debug SA {scene_folder}]: Found {len(sample_token_map)} samples for this scene.") # DEBUG
        if not sample_token_map:
            # print(f"[Debug SA {scene_folder}]: No samples in sample_token_map, exiting annotation generation for this scene.") # DEBUG
            return

        instance_annotations = {} 
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
                        
                        quaternion = euler_to_quaternion(
                            float(rotation_data.get("roll", 0)),
                            float(rotation_data.get("pitch", 0)),
                            float(rotation_data.get("yaw", 0))
                        )
                        
                        velocity = annotation_data.get("velocity", {})
                        velocity_magnitude = velocity.get("magnitude", 0.0)
                        vehicle_state = self._infer_vehicle_state(velocity_magnitude)
                        attribute_token = self.converter.attribute_name_to_token.get(vehicle_state, "")
                        
                        box_center = [
                            float(translation_data.get("x", 0)),
                            float(translation_data.get("y", 0)),
                            float(translation_data.get("z", 0))
                        ]
                        box_size = annotation_data.get("size", [0,0,0]) # This should be NuScenes format [w,l,h]
                                                
                        num_lidar_pts = 0
                        # Assuming Lidar data is in a folder named as per sim_config sensor name
                        # This part relies on self.converter._get_sensor_transform which is in NuScenesConverter
                        lidar_sensor_name = next((s["name"] for s in self.converter.sim_config.get("sensors", []) if s["type"] == "lidar"), "Lidar") # Default or find
                        lidar_file = scene_path / lidar_sensor_name / f"{timestamp}.npy"
                        if lidar_file.exists():
                            try:
                                lidar_points = np.load(lidar_file)
                                sensor_transform_rel, ego_transform_abs = self.converter._get_sensor_transform(scene_folder, lidar_sensor_name, timestamp)
                                if sensor_transform_rel and ego_transform_abs:
                                    global_points = transform_points_to_global(lidar_points, sensor_transform_rel, ego_transform_abs)
                                    num_lidar_pts = count_points_in_box(global_points, box_center, box_size, quaternion)
                            except Exception as e:
                                print(f"Warning: Error processing lidar points for annotation: {e}")
                        
                        num_radar_pts = 0
                        radar_sensor_configs = [s for s in self.converter.sim_config.get("sensors", []) if s["type"] == "radar"]
                        for radar_cfg in radar_sensor_configs:
                            radar_name = radar_cfg["name"]
                            radar_file = scene_path / radar_name / f"{timestamp}.npy"
                            if radar_file.exists():
                                try:
                                    radar_points = np.load(radar_file)
                                    sensor_transform_rel, ego_transform_abs = self.converter._get_sensor_transform(scene_folder, radar_name, timestamp)
                                    if sensor_transform_rel and ego_transform_abs:
                                        global_points = transform_radar_points_to_global(radar_points, sensor_transform_rel, ego_transform_abs)
                                        if len(global_points) > 0:
                                            num_radar_pts += count_points_in_box(global_points, box_center, box_size, quaternion)
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
import json
from nuscene_utils import generate_token

class InstanceGenerator:
    def __init__(self, converter):
        self.converter = converter

    def generate_instance_entries(self, scene_folder: str, scene_token: str):
        """Generate instance.json entries for a scene based on 3dbbox annotations.
        
        Args:
            scene_folder: Name of the scene folder
            scene_token: Token of the scene
        """
        # Get category token mapping
        category_mappings = self.converter.config.get("category_mappings", {})
        category_name_to_token = {entry["name"]: entry["token"] for entry in self.converter.categories}
        
        # Track actor -> category (prefer specific categories from bbox 'category')
        actor_to_category = {}
        
        # Find all 3dbbox JSON files in the scene
        scene_path = self.converter.input_base / scene_folder
        for sensor_folder in scene_path.iterdir():
            if not sensor_folder.is_dir():
                continue
                
            bbox_files = list(sensor_folder.glob("*_3dbbox.json"))
            for bbox_file in bbox_files:
                try:
                    if bbox_file.stat().st_size <= 2:  # Skip empty files
                        continue
                        
                    with open(bbox_file, 'r') as f:
                        bbox_data = json.load(f)
                        
                    for annotation in bbox_data:
                        actor_id = annotation.get("actor_id")
                        # Prefer explicit category from bbox export; fallback to coarse 'type'
                        ann_category = annotation.get("category")
                        if not ann_category:
                            ann_type = annotation.get("type")
                            if ann_type == "vehicle":
                                ann_category = "vehicle.car"  # reasonable default
                            elif ann_type == "pedestrian":
                                ann_category = "human.pedestrian.adult"
                        if actor_id is not None and ann_category is not None:
                            # First non-empty category wins; assumes consistent category per actor
                            actor_to_category.setdefault(actor_id, ann_category)
                            
                except Exception as e:
                    print(f"Error reading bbox file {bbox_file}: {e}")
                    continue
        
        for actor_id, ann_category in actor_to_category.items():
            # Map bbox category to target nuScenes category using converter_config
            nuscene_category = category_mappings.get(ann_category) or category_mappings.get(ann_category.split('.')[0])
            if not nuscene_category:
                print(f"Warning: No category mapping found for category key '{ann_category}'")
                continue
                
            category_token = category_name_to_token.get(nuscene_category)
            if not category_token:
                print(f"Warning: No category token found for NuScenes category {nuscene_category}")
                continue
            
            instance_token = generate_token()
            instance_entry = {
                "token": instance_token,
                "category_token": category_token,
                "nbr_annotations": None, 
                "first_annotation_token": None,
                "last_annotation_token": None
            }
            
            self.converter.token_maps['instance'][(scene_folder, actor_id)] = instance_token
            self.converter.instances.append(instance_entry)

    def update_instance_entries(self):
        """Update instance entries with annotation counts and first/last annotation tokens."""
        sample_token_to_timestamp = {sample["token"]: sample["timestamp"] for sample in self.converter.samples}
        
        instance_annotations = {}
        for annotation in self.converter.sample_annotations:
            instance_token = annotation["instance_token"]
            if instance_token not in instance_annotations:
                instance_annotations[instance_token] = []
            instance_annotations[instance_token].append(annotation)
        
        for instance in self.converter.instances:
            instance_token = instance["token"]
            annotations = instance_annotations.get(instance_token, [])
            
            if annotations:
                annotations.sort(key=lambda x: sample_token_to_timestamp.get(x["sample_token"], 0))
                
                instance["nbr_annotations"] = len(annotations)
                instance["first_annotation_token"] = annotations[0]["token"]
                instance["last_annotation_token"] = annotations[-1]["token"]
            else:
                instance["nbr_annotations"] = 0
                instance["first_annotation_token"] = ""
                instance["last_annotation_token"] = "" 
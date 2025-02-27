import os
import yaml
import glob
from pathlib import Path
import cv2
import numpy as np
import json

def detect_vehicle_instance_boxes(image_path, vehicle_tags=[14, 15, 16, 18]):
    """
    Detect bounding boxes of vehicles from an instance segmentation image.
    Vehicle tags:
    - 14: Car
    - 15: Truck
    - 16: Bus
    - 18: Motorcycle
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image from {image_path}")
        return [], []
    
    # Split channels (BGR)
    b, g, r = cv2.split(image)
    
    # Create vehicle mask for all vehicle types
    vehicle_mask = np.zeros_like(r, dtype=bool)
    for tag in vehicle_tags:
        vehicle_mask |= (r == tag)
    
    # Find all unique instance IDs
    instance_ids = {}  # Dict to map instance ID to pixels
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if vehicle_mask[y, x]:
                # Calculate unique instance ID from green and blue values
                instance_id = (int(g[y, x]) << 8) | int(b[y, x])
                if instance_id not in instance_ids:
                    instance_ids[instance_id] = []
                instance_ids[instance_id].append((x, y))
    
    # Find bounding box for each vehicle instance
    bounding_boxes = []
    ids = []
    
    for instance_id, pixels in instance_ids.items():
        if len(pixels) > 50:  # Minimum size threshold to filter noise
            pixels = np.array(pixels)
            x_min = np.min(pixels[:, 0])
            y_min = np.min(pixels[:, 1])
            x_max = np.max(pixels[:, 0])
            y_max = np.max(pixels[:, 1])
            
            width = x_max - x_min
            height = y_max - y_min
            
            bounding_boxes.append((int(x_min), int(y_min), int(width), int(height)))
            ids.append(instance_id)
    
    return bounding_boxes, ids

def find_paired_instance_image(rgb_image_path, instance_folder):
    """Find the corresponding instance segmentation image"""
    timestamp = Path(rgb_image_path).stem
    instance_path = Path(instance_folder) / f"{timestamp}.png"
    return str(instance_path) if instance_path.exists() else None

def get_camera_config(camera_name, config):
    """Get camera parameters from config file"""
    for sensor in config["sensors"]:
        if sensor["name"] == camera_name and sensor["type"] == "camera":
            return {
                "width": int(sensor["attributes"]["image_size_x"]),
                "height": int(sensor["attributes"]["image_size_y"]),
                "fov": float(sensor["attributes"].get("fov", 90)) 
            }
    return None

def process_scene(scene_path):
    """Process all RGB cameras in a scene that have instance segmentation pairs"""
    # Load config file to get camera parameters
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Processing scene: {scene_path}")
    
    # Get all camera folders
    camera_folders = []
    for folder in os.listdir(scene_path):
        folder_path = os.path.join(scene_path, folder)
        if os.path.isdir(folder_path) and folder.startswith("camera_"):
            camera_folders.append(folder_path)
    
    # Process each camera
    for camera_folder in camera_folders:
        camera_name = os.path.basename(camera_folder)
        instance_folder = os.path.join(scene_path, f"instance_{camera_name}")
        
        # Get camera configuration
        camera_config = get_camera_config(camera_name, config)
        if not camera_config:
            print(f"Warning: No configuration found for camera {camera_name}")
            continue
        
        if not os.path.exists(instance_folder):
            continue
            
        print(f"Processing {camera_name}", end="... ")
        
        # Process each RGB image
        rgb_images = glob.glob(os.path.join(camera_folder, "*.png"))
        processed = 0
        for rgb_path in rgb_images:
            instance_path = find_paired_instance_image(rgb_path, instance_folder)
            if not instance_path:
                continue
                
            try:
                boxes, instance_ids = detect_vehicle_instance_boxes(instance_path)
                
                # Save bounding box data using camera configuration
                timestamp = Path(rgb_path).stem
                bbox_data = {
                    "image_file": f"{timestamp}.png",
                    "timestamp": timestamp,
                    "camera_data": camera_config,
                    "bounding_boxes": [
                        {
                            "vehicle_id": instance_id,
                            "bbox": list(map(float, box))
                        }
                        for box, instance_id in zip(boxes, instance_ids)
                    ]
                }
                
                json_path = os.path.join(camera_folder, f"{timestamp}_bbox.json")
                with open(json_path, 'w') as f:
                    json.dump(bbox_data, f, indent=2)
                processed += 1
                
            except Exception as e:
                print(f"\nError processing {rgb_path}: {e}")
        print(f"done ({processed} frames)")

def main():
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    base_save_path = config["simulation"]["base_save_path"]
    
    scene_dirs = sorted(glob.glob(os.path.join(base_save_path, "scene_*")))
    for scene_dir in scene_dirs:
        process_scene(scene_dir)

if __name__ == "__main__":
    main()

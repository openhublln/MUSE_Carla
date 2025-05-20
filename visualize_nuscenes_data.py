import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
import os
import json

# Another venv is needed for this script because of the nuscenes-devkit dependency

# --- Configuration ---
# Replace with the actual path to your nuScenes data
NUSCENES_DATA_ROOT = './nuscenes_format'
# Using custom format without version directory
NUSCENES_VERSION = 'v1.0-mini'  # Changed to match official format
# --- End Configuration ---

def validate_data_files():
    """Validate that all required data files exist and are properly formatted."""
    required_files = [
        'sample.json',
        'sample_data.json',
        'ego_pose.json',
        'scene.json',
        'sensor.json',
        'calibrated_sensor.json'
    ]
    
    version_dir = os.path.join(NUSCENES_DATA_ROOT, NUSCENES_VERSION)
    if not os.path.exists(version_dir):
        print(f"Error: Version directory not found: {version_dir}")
        return False
        
    for file in required_files:
        file_path = os.path.join(version_dir, file)
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            return False
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    print(f"Error: {file} does not contain a JSON array")
                    return False
                if len(data) == 0:
                    print(f"Warning: {file} is empty")
        except json.JSONDecodeError:
            print(f"Error: {file} is not valid JSON")
            return False
        except Exception as e:
            print(f"Error reading {file}: {e}")
            return False
            
    return True

def validate_data_relationships():
    """Validate that the relationships between data tables are correct."""
    version_dir = os.path.join(NUSCENES_DATA_ROOT, NUSCENES_VERSION)
    
    print("\nValidating data relationships...")
    
    # Load all required JSON files
    with open(os.path.join(version_dir, 'scene.json'), 'r') as f:
        scenes = json.load(f)
    with open(os.path.join(version_dir, 'sample.json'), 'r') as f:
        samples = json.load(f)
    with open(os.path.join(version_dir, 'sample_data.json'), 'r') as f:
        sample_data = json.load(f)
    with open(os.path.join(version_dir, 'sensor.json'), 'r') as f:
        sensors = json.load(f)
    with open(os.path.join(version_dir, 'calibrated_sensor.json'), 'r') as f:
        calibrated_sensors = json.load(f)
    
    print(f"Loaded {len(scenes)} scenes, {len(samples)} samples, {len(sample_data)} sample_data entries")
    
    # Create lookup dictionaries
    sample_tokens = {s['token']: s for s in samples}
    sensor_tokens = {s['token']: s for s in sensors}
    calibrated_sensor_tokens = {cs['token']: cs for cs in calibrated_sensors}
    
    # Validate camera intrinsics
    print("\nValidating camera intrinsics...")
    for cs in calibrated_sensors:
        sensor = sensor_tokens.get(cs['sensor_token'])
        if sensor and sensor['modality'] == 'camera':
            if not cs.get('camera_intrinsic') or len(cs['camera_intrinsic']) == 0:
                print(f"Warning: Camera sensor {sensor['channel']} has no camera intrinsics")
                # Add default camera intrinsics
                cs['camera_intrinsic'] = [
                    [800.0, 0.0, 400.0],  # fx, 0, cx
                    [0.0, 800.0, 300.0],  # 0, fy, cy
                    [0.0, 0.0, 1.0]       # 0, 0, 1
                ]
                # Write back to file
                with open(os.path.join(version_dir, 'calibrated_sensor.json'), 'w') as f:
                    json.dump(calibrated_sensors, f, indent=2)
    
    # Validate scene-sample relationships
    for scene in scenes:
        print(f"\nValidating scene {scene['token']} ({scene['name']})")
        
        if not scene.get('first_sample_token'):
            print(f"Warning: Scene {scene['token']} has no first_sample_token")
            continue
            
        # Find the sample
        sample = sample_tokens.get(scene['first_sample_token'])
        if not sample:
            print(f"Error: Scene {scene['token']} references non-existent sample {scene['first_sample_token']}")
            return False
            
        print(f"Found first sample {sample['token']} with timestamp {sample['timestamp']}")
        
        # Validate sample-sample_data relationships
        sample_data_entries = [sd for sd in sample_data if sd['sample_token'] == sample['token']]
        print(f"Found {len(sample_data_entries)} sample_data entries for this sample")
        
        for sample_d in sample_data_entries:
            # Check if the file exists
            file_path = os.path.join(NUSCENES_DATA_ROOT, sample_d['filename'])
            if not os.path.exists(file_path):
                print(f"Error: Sample data file not found: {file_path}")
                return False
                
            # Validate sensor relationships
            if sample_d.get('calibrated_sensor_token'):
                calibrated_sensor = calibrated_sensor_tokens.get(sample_d['calibrated_sensor_token'])
                if not calibrated_sensor:
                    print(f"Error: Sample data {sample_d['token']} references non-existent calibrated sensor {sample_d['calibrated_sensor_token']}")
                    return False
                    
                sensor = sensor_tokens.get(calibrated_sensor['sensor_token'])
                if not sensor:
                    print(f"Error: Calibrated sensor {calibrated_sensor['token']} references non-existent sensor {calibrated_sensor['sensor_token']}")
                    return False
                    
                # Validate camera intrinsics for camera sensors
                if sensor['modality'] == 'camera' and (not calibrated_sensor.get('camera_intrinsic') or len(calibrated_sensor['camera_intrinsic']) == 0):
                    print(f"Error: Camera sensor {sensor['channel']} has no camera intrinsics")
                    return False
    
    print("\nData relationship validation completed successfully")
    return True

def main():
    """
    Main function to load nuScenes data and visualize a sample.
    """
    print(f"Initializing NuScenes dataset with version: {NUSCENES_VERSION} and dataroot: {NUSCENES_DATA_ROOT}")
    print("This might take a few moments...")

    # First validate the data files
    print("\nStep 1: Validating data files...")
    if not validate_data_files():
        print("\nData validation failed. Please check the errors above.")
        return
        
    # Then validate data relationships
    print("\nStep 2: Validating data relationships...")
    if not validate_data_relationships():
        print("\nData relationship validation failed. Please check the errors above.")
        return

    print("\nStep 3: Attempting to initialize NuScenes dataset...")
    try:
        # Print the contents of a few key files for debugging
        version_dir = os.path.join(NUSCENES_DATA_ROOT, NUSCENES_VERSION)
        print("\nChecking key files:")
        
        # Check scene.json
        scene_path = os.path.join(version_dir, 'scene.json')
        if os.path.exists(scene_path):
            with open(scene_path, 'r') as f:
                scenes = json.load(f)
                print(f"\nFirst scene entry: {json.dumps(scenes[0], indent=2)}")
        
        # Check sample.json
        sample_path = os.path.join(version_dir, 'sample.json')
        if os.path.exists(sample_path):
            with open(sample_path, 'r') as f:
                samples = json.load(f)
                print(f"\nFirst sample entry: {json.dumps(samples[0], indent=2)}")
        
        # Check sample_data.json
        sample_data_path = os.path.join(version_dir, 'sample_data.json')
        if os.path.exists(sample_data_path):
            with open(sample_data_path, 'r') as f:
                sample_data = json.load(f)
                print(f"\nFirst sample_data entry: {json.dumps(sample_data[0], indent=2)}")
                
        # Check sensor.json
        sensor_path = os.path.join(version_dir, 'sensor.json')
        if os.path.exists(sensor_path):
            with open(sensor_path, 'r') as f:
                sensors = json.load(f)
                print(f"\nAll sensor entries: {json.dumps(sensors, indent=2)}")
                
        # Check calibrated_sensor.json
        calibrated_sensor_path = os.path.join(version_dir, 'calibrated_sensor.json')
        if os.path.exists(calibrated_sensor_path):
            with open(calibrated_sensor_path, 'r') as f:
                calibrated_sensors = json.load(f)
                print(f"\nAll calibrated sensor entries: {json.dumps(calibrated_sensors, indent=2)}")
        
        nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATA_ROOT, verbose=True)
        print("\nNuScenes dataset initialized successfully.")
    except Exception as e:
        print(f"\nError initializing NuScenes dataset: {e}")
        print("Please ensure that:")
        print(f"1. The NUSCENES_DATA_ROOT ('{NUSCENES_DATA_ROOT}') is correct and points to your nuScenes root directory.")
        print(f"2. The NUSCENES_VERSION ('{NUSCENES_VERSION}') is correct (e.g., 'v1.0-mini', 'v1.0-trainval').")
        print("3. The nuScenes devkit is installed correctly (pip install nuscenes-devkit).")
        print("4. Your dataset directory structure matches the expected nuScenes format (e.g., samples, sweeps, maps, v1.0-mini folders).")
        return

    if not nusc.scene:
        print("\nNo scenes found in the dataset. Please check your data and configuration.")
        return

    print(f"\nTotal number of scenes: {len(nusc.scene)}")
    my_scene = nusc.scene[0]
    print(f"Working with scene: {my_scene['name']} (token: {my_scene['token']})")

    first_sample_token = my_scene['first_sample_token']
    if not first_sample_token:
        print("Error: No first sample token found in scene")
        return
        
    try:
        my_sample = nusc.get('sample', first_sample_token)
        print(f"First sample token: {first_sample_token}")
    except Exception as e:
        print(f"Error getting sample data: {e}")
        return

    # 1. Render the entire sample (all sensor data and annotations)
    print("\nRendering sample... (Close the plot window to continue)")
    try:
        nusc.render_sample(first_sample_token)
        plt.show() # Ensures the plot is displayed
    except Exception as e:
        print(f"Error during nusc.render_sample: {e}")

    # 2. Render LIDAR point cloud in the front camera image
    print("\nRendering LIDAR point cloud in front camera image... (Close the plot window to continue)")
    if 'LIDAR_TOP' in my_sample['data'] and 'CAM_FRONT' in my_sample['data']:
        try:
            nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP', camera_channel='CAM_FRONT')
            plt.show()
        except Exception as e:
            print(f"Error during nusc.render_pointcloud_in_image: {e}")
    else:
        print("LIDAR_TOP or CAM_FRONT data not available for this sample.")

    # 3. Render a specific sample_data (e.g., front camera image with annotations)
    cam_front_token = my_sample['data'].get('CAM_FRONT')
    if cam_front_token:
        print(f"\nRendering CAM_FRONT data (token: {cam_front_token})... (Close the plot window to continue)")
        try:
            nusc.render_sample_data(cam_front_token, with_anns=True)
            plt.show()
        except Exception as e:
            print(f"Error during nusc.render_sample_data for CAM_FRONT: {e}")
    else:
        print("CAM_FRONT data not available for this sample.")

    # 4. Render a scene as a video (optional, can be slow and resource-intensive)
    # This will save a video file in your current working directory.
    # OpenCV is required for this (pip install opencv-python).
    # Hit ESC to stop the video rendering.
    render_scene_video = False # Set to True to enable
    if render_scene_video:
        print(f"\nRendering scene '{my_scene['name']}' as a video for CAM_FRONT channel...")
        print("This may take some time. Press ESC in the video window to stop.")
        output_video_path = f"./{my_scene['name']}_CAM_FRONT.avi"
        try:
            nusc.render_scene_channel(my_scene['token'], 'CAM_FRONT', out_path=output_video_path)
            print(f"Scene video saved to: {os.path.abspath(output_video_path)}")
        except Exception as e:
            print(f"Error during nusc.render_scene_channel: {e}")
            print("Ensure OpenCV is installed (pip install opencv-python) and you have a display environment if not running headless.")
    else:
        print("\nScene video rendering is disabled. Set 'render_scene_video = True' in the script to enable it.")

    print("\nVisualization script finished.")

if __name__ == '__main__':
    main() 
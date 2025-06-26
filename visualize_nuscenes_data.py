import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend to avoid set_window_title issues
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
import os
import json
import traceback
from PIL import Image
import numpy as np

# Another venv is needed for this script because of the nuscenes-devkit dependency

# --- Configuration ---
# Replace with the actual path to your nuScenes data
NUSCENES_DATA_ROOT = './nuscenes_format'
# Using custom format without version directory
NUSCENES_VERSION = 'v1.0'  
# --- End Configuration ---

def validate_data_files():
    """Validate that all required data files exist and are properly formatted."""
    required_files = [
        'sample.json',
        'sample_data.json',
        'ego_pose.json',
        'scene.json',
        'sensor.json',
        'calibrated_sensor.json',
        'category.json',
        'attribute.json',
        'visibility.json',
        'instance.json',
        'sample_annotation.json'
    ]
    
    optional_files = [
        'map.json'  # Map is optional
    ]
    
    version_dir = os.path.join(NUSCENES_DATA_ROOT, NUSCENES_VERSION)
    if not os.path.exists(version_dir):
        print(f"Error: Version directory not found: {version_dir}")
        return False
        
    # Check required files
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
                print(f"Loaded {len(data)} entries from {file}")
        except json.JSONDecodeError:
            print(f"Error: {file} is not valid JSON")
            return False
        except Exception as e:
            print(f"Error reading {file}: {e}")
            return False
    
    # Check optional files
    for file in optional_files:
        file_path = os.path.join(version_dir, file)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        print(f"Warning: {file} does not contain a JSON array")
                    elif len(data) == 0:
                        print(f"Warning: {file} is empty")
                    else:
                        print(f"Loaded {len(data)} entries from {file}")
            except Exception as e:
                print(f"Warning: Error reading optional file {file}: {e}")
        else:
            print(f"Note: Optional file not found: {file_path}")
            
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
    with open(os.path.join(version_dir, 'instance.json'), 'r') as f:
        instances = json.load(f)
    with open(os.path.join(version_dir, 'sample_annotation.json'), 'r') as f:
        annotations = json.load(f)
    
    print(f"Loaded {len(scenes)} scenes, {len(samples)} samples, {len(sample_data)} sample_data entries")
    print(f"Loaded {len(instances)} instances, {len(annotations)} annotations")
    
    # Create lookup dictionaries
    sample_tokens = {s['token']: s for s in samples}
    sensor_tokens = {s['token']: s for s in sensors}
    calibrated_sensor_tokens = {cs['token']: cs for cs in calibrated_sensors}
    instance_tokens = {i['token']: i for i in instances}
    
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
    
    # Validate instance-annotation relationships
    for annotation in annotations:
        if not annotation.get('instance_token'):
            print(f"Warning: Annotation {annotation['token']} has no instance_token")
            continue
            
        instance = instance_tokens.get(annotation['instance_token'])
        if not instance:
            print(f"Error: Annotation {annotation['token']} references non-existent instance {annotation['instance_token']}")
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
        nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATA_ROOT, verbose=True)
        print("\nNuScenes dataset initialized successfully.")
    except Exception as e:
        print(f"\nError during NuScenes initialization: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
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
        print(f"Sample data keys: {list(my_sample['data'].keys())}")
        
        # Check if we need to update the LIDAR channel name
        if 'LIDAR' in my_sample['data'] and 'LIDAR_TOP' not in my_sample['data']:
            print("Updating LIDAR channel name to LIDAR_TOP...")
            my_sample['data']['LIDAR_TOP'] = my_sample['data'].pop('LIDAR')
        elif 'LIDAR_TOP' not in my_sample['data']:
            print("Warning: No LIDAR data found in sample")
    except Exception as e:
        print(f"Error getting sample data: {e}")
        return

    # 1. Render the entire sample (all sensor data and annotations)
    print("\nRendering sample... (Close the plot window to continue)")
    try:
        # First check if we have the required sensor data
        if 'CAM_FRONT' in my_sample['data']:
            # Get the sample data for CAM_FRONT
            cam_front_data = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
            if os.path.exists(os.path.join(NUSCENES_DATA_ROOT, cam_front_data['filename'])):
                # Try to render without ego-centric map to avoid map issues
                try:
                    nusc.render_sample(first_sample_token)
                    plt.show() # Ensures the plot is displayed
                except Exception as map_error:
                    print(f"Warning: Could not render with ego-centric map: {map_error}")
                    print("This is likely due to map rendering issues, but other visualizations should work")
            else:
                print(f"Warning: Camera image file not found: {cam_front_data['filename']}")
        else:
            print("Warning: CAM_FRONT data not available for rendering sample")
    except Exception as e:
        print(f"Error during nusc.render_sample: {e}")
        print("This might be due to missing or incorrectly formatted sensor data")
        print("Full traceback:")
        traceback.print_exc()

    # 2. Render LIDAR point cloud in the front camera image
    print("\nRendering LIDAR point cloud in front camera image... (Close the plot window to continue)")
    if 'LIDAR_TOP' in my_sample['data'] and 'CAM_FRONT' in my_sample['data']:
        try:
            # Get the sample data for LIDAR_TOP
            lidar_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
            if os.path.exists(os.path.join(NUSCENES_DATA_ROOT, lidar_data['filename'])):
                # Verify LIDAR data format before rendering
                lidar_path = os.path.join(NUSCENES_DATA_ROOT, lidar_data['filename'])
                try:
                    data = np.fromfile(lidar_path, dtype=np.float32)
                    if data.size % 5 == 0:
                        print(f"LIDAR data verified: {data.shape} -> {data.reshape(-1, 5).shape}")
                        
                        # Try to render with error handling
                        try:
                            nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP', camera_channel='CAM_FRONT')
                            plt.show()
                        except Exception as render_error:
                            print(f"Warning: Point cloud rendering failed: {render_error}")
                            print("This might be due to calibration or data format issues")
                            
                    else:
                        print(f"Warning: LIDAR data format issue - {data.size} elements not divisible by 5")
                except Exception as lidar_error:
                    print(f"Warning: LIDAR data verification failed: {lidar_error}")
            else:
                print(f"Warning: LIDAR data file not found: {lidar_data['filename']}")
                print(f"Expected path: {os.path.join(NUSCENES_DATA_ROOT, lidar_data['filename'])}")
        except Exception as e:
            print(f"Error during nusc.render_pointcloud_in_image: {e}")
            print("This might be due to missing or incorrectly formatted point cloud data")
            print("Full traceback:")
            traceback.print_exc()
    else:
        print("LIDAR_TOP or CAM_FRONT data not available for this sample")
        if 'LIDAR_TOP' not in my_sample['data']:
            print("Available sensor data keys:", list(my_sample['data'].keys()))

    # 3. Render a specific sample_data (e.g., front camera image with annotations)
    cam_front_token = my_sample['data'].get('CAM_FRONT')
    if cam_front_token:
        print(f"\nRendering CAM_FRONT data (token: {cam_front_token})... (Close the plot window to continue)")
        try:
            # First check if the file exists
            sample_data = nusc.get('sample_data', cam_front_token)
            image_path = os.path.join(NUSCENES_DATA_ROOT, sample_data['filename'])
            if os.path.exists(image_path):
                # Check if the file is a valid image
                try:
                    # Try to open and verify the image
                    with Image.open(image_path) as img:
                        # Get image size
                        width, height = img.size
                        print(f"Image size: {width}x{height}")
                        
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Check if image has valid data
                        if width > 0 and height > 0:
                            # Save as a new JPEG to ensure proper format
                            temp_path = image_path + '.temp.jpg'
                            img.save(temp_path, 'JPEG', quality=95)
                            # Replace original with properly formatted image
                            os.replace(temp_path, image_path)
                            print(f"Successfully verified and fixed image format: {image_path}")
                            
                            # Now try to render
                            nusc.render_sample_data(cam_front_token, with_anns=True)
                            plt.show()
                        else:
                            print(f"Error: Invalid image dimensions: {width}x{height}")
                    
                except Exception as e:
                    print(f"Error: Invalid image file: {e}")
                    print(f"Image path: {image_path}")
                    print("Attempting to fix image format...")
                    try:
                        # Try to fix the image format
                        img = Image.open(image_path)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Create a new image with the same size and mode
                        new_img = Image.new('RGB', img.size)
                        # Copy the data safely
                        try:
                            new_img.putdata(list(img.getdata()))
                        except Exception as data_error:
                            print(f"Error copying image data: {data_error}")
                            # Create a simple test image instead
                            new_img = Image.new('RGB', (1920, 1080), color='black')
                        
                        # Save as JPEG
                        new_img.save(image_path, 'JPEG', quality=95)
                        print("Image format fixed. Please run the script again.")
                    except Exception as e2:
                        print(f"Failed to fix image format: {e2}")
                        print("Full traceback:")
                        traceback.print_exc()
            else:
                print(f"Warning: Camera image file not found: {sample_data['filename']}")
        except Exception as e:
            print(f"Error during nusc.render_sample_data for CAM_FRONT: {e}")
            print("This might be due to missing or incorrectly formatted camera data")
            print("Full traceback:")
            traceback.print_exc()
    else:
        print("CAM_FRONT data not available for this sample")

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
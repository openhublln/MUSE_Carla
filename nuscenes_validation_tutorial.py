# nuScenes Validation Tutorial Script
# Follows: https://colab.research.google.com/github/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/nuscenes_tutorial.ipynb

import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from nuscenes.utils import splits
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.config import config_factory as det_config_factory
# from nuscenes.eval.detection.render import render_sample_result
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

# 1. Set up the dataset path and version
DATASET_PATH = os.path.abspath('nuscenes_format') 
VERSION = 'v1.0' 

# 2. Load the nuScenes dataset
print(f"Loading nuScenes dataset from: {DATASET_PATH}, version: {VERSION}")
nusc = NuScenes(version=VERSION, dataroot=DATASET_PATH, verbose=True)

# 3. Explore the dataset
print(f"Number of scenes: {len(nusc.scene)}")
print(f"Number of samples: {len(nusc.sample)}")
print(f"Number of annotations: {len(nusc.sample_annotation)}")

# 4. Visualize a sample (first scene, first sample)
scene = nusc.scene[0]
first_sample_token = scene['first_sample_token']
sample = nusc.get('sample', first_sample_token)

print(f"\nScene name: {scene['name']}")
print(f"Sample token: {first_sample_token}")

# List all camera data in this sample
print("\nCamera data in this sample:")
for sensor in sample['data']:
    if 'CAM' in sensor:
        print(f"  {sensor}: {sample['data'][sensor]}")

# Check if annotations exist for the sample
sample_token = sample['token']
anns = [ann for ann in nusc.sample_annotation if ann['sample_token'] == sample_token]
print(f"\nAnnotations for sample {sample_token}:")
if anns:
    for ann in anns:
        print(f"  token: {ann['token']}, translation: {ann['translation']}, category: {ann.get('category_name', 'N/A')}")
else:
    print("  No annotations found for this sample.")

# Print camera calibration and ego pose for CAM_BACKLEFT
if 'CAM_BACKLEFT' in sample['data']:
    cam_token = sample['data']['CAM_BACKLEFT']
    cam_sd = nusc.get('sample_data', cam_token)
    calib = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    ego = nusc.get('ego_pose', cam_sd['ego_pose_token'])
    print(f"\nCAM_BACKLEFT calibration (translation): {calib['translation']}")
    print(f"CAM_BACKLEFT calibration (rotation): {calib['rotation']}")
    print(f"CAM_BACKLEFT ego pose (translation): {ego['translation']}")
    print(f"CAM_BACKLEFT ego pose (rotation): {ego['rotation']}")

# Visualize camera image with boxes (CAM_BACKLEFT if available)
if 'CAM_BACKLEFT' in sample['data']:
    nusc.render_sample_data(sample['data']['CAM_BACKLEFT'], with_anns=True)
    plt.show()
else:
    print("CAM_BACKLEFT not found in this sample.")

# 5. Visualize LIDAR_TOP point cloud with boxes
if 'LIDAR_TOP' in sample['data']:
    nusc.render_sample_data(sample['data']['LIDAR_TOP'], nsweeps=1, underlay_map=False, with_anns=True)
    plt.show()
else:
    print("LIDAR_TOP not found in this sample.")

# 6. Print all categories
print("\nCategories:")
for cat in nusc.category:
    print(f"  {cat['name']}")

# 7. Print all attribute names
print("\nAttributes:")
for attr in nusc.attribute:
    print(f"  {attr['name']}")

# 8. Print all visibility levels
print("\nVisibility levels:")
for vis in nusc.visibility:
    print(f"  {vis['level']} - {vis['description']}")

# 9. Print all sensor modalities
print("\nSensors:")
for sensor in nusc.sensor:
    print(f"  {sensor['channel']} ({sensor['modality']})")

# 10. Print all calibrated sensors
print("\nCalibrated Sensors:")
for cs in nusc.calibrated_sensor:
    print(f"  {cs['token']} - {cs['translation']} {cs['rotation']}")

# 11. Print all logs
print("\nLogs:")
for log in nusc.log:
    print(f"  {log['token']} - {log['logfile']}")

# 12. Print all maps
print("\nMaps:")
for m in nusc.map:
    print(f"  {m['token']} - {m['filename']}")

# 13. Print a summary of a sample annotation
if len(nusc.sample_annotation) > 0:
    ann = nusc.sample_annotation[0]
    print("\nSample annotation example:")
    print(ann)
else:
    print("No sample annotations found.")
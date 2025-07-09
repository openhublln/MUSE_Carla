"""
Minimal nuScenes visualizer for CARLA-converted dataset.
Step 3: Display the LIDAR point cloud with 3D bounding box annotations from sample_annotation.json.
"""

import os
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Path to your converted nuScenes-format dataset
NUSCENES_ROOT = os.path.join(os.path.dirname(__file__), 'nuscenes_format')
VERSION = 'v1.0'  # Change if you used a different version name

# Initialize nuScenes object
nusc = NuScenes(version=VERSION, dataroot=NUSCENES_ROOT, verbose=True)

# Get the first scene
scene = nusc.scene[0]
print(f"Loaded scene: {scene['name']}")

# Get the first sample in the scene
first_sample_token = scene['first_sample_token']
sample = nusc.get('sample', first_sample_token)
print(f"First sample token: {first_sample_token}")

# Find the LIDAR_TOP sample_data token for this sample using the standard nuScenes API
lidar_channel = 'LIDAR_TOP'
sample_data_token = None
for sd in nusc.sample_data:
    if sd['sample_token'] == sample['token'] and sd['channel'] == lidar_channel:
        sample_data_token = sd['token']
        break

if sample_data_token is None:
    raise RuntimeError("No LIDAR_TOP sample_data found for the first sample.")

sd = nusc.get('sample_data', sample_data_token)
lidar_path = os.path.join(NUSCENES_ROOT, sd['filename'])
print(f"Displaying LIDAR point cloud: {lidar_path}")

# Load the LIDAR point cloud (nuScenes format: float32, [N,5]: x, y, z, intensity, ring_index)
points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)

# Get all bounding boxes for this sample
boxes = nusc.get_boxes(sample_data_token)

# Get the ego pose for this sample
ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])
print(f"Ego pose translation: {ego_pose['translation']}")

# Get the LIDAR sensor calibration for this sample
def get_lidar_calib(nusc, sd):
    calib_token = sd['calibrated_sensor_token']
    calib = nusc.get('calibrated_sensor', calib_token)
    print(f"LIDAR calibration translation (sensor->ego): {calib['translation']}")
    print(f"LIDAR calibration rotation (sensor->ego, quaternion [w,x,y,z]): {calib['rotation']}")
    return calib

lidar_calib = get_lidar_calib(nusc, sd)

# Diagnostic print: show box sizes and centers
for i, box in enumerate(boxes):
    # Try to get yaw in radians (heading)
    try:
        # Some versions of nuScenes Box have .yaw, others have .orientation.yaw_pitch_roll
        if hasattr(box, 'yaw'):
            yaw = box.yaw
        elif hasattr(box.orientation, 'yaw_pitch_roll'):
            yaw = box.orientation.yaw_pitch_roll[0]
        else:
            yaw = None
    except Exception:
        yaw = None
    print(f"Box {i}: center={box.center}, size(wlh)={box.wlh}, quaternion={box.orientation.q}, yaw(rad)={yaw}")

# Plot the point cloud (x, y, z) in 3D and top-down (XY) views only
plt.ion()
fig = plt.figure(figsize=(14, 7))

# 3D view
ax3d = fig.add_subplot(121, projection='3d')
ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.2, c=points[:, 2], cmap='viridis', alpha=0.5)
colors = plt.cm.get_cmap('tab20', len(boxes))
for i, box in enumerate(boxes):
    corners = box.corners().T
    faces = [
        [corners[j] for j in [0, 1, 2, 3]],
        [corners[j] for j in [4, 5, 6, 7]],
        [corners[j] for j in [0, 1, 5, 4]],
        [corners[j] for j in [2, 3, 7, 6]],
        [corners[j] for j in [1, 2, 6, 5]],
        [corners[j] for j in [4, 7, 3, 0]],
    ]
    box_color = colors(i)[:3]
    pc = Poly3DCollection(faces, facecolors=[box_color], linewidths=0.8, edgecolors=[box_color], alpha=0.25)
    ax3d.add_collection3d(pc)
    for face in faces:
        xs, ys, zs = zip(*face)
        ax3d.plot(xs + (xs[0],), ys + (ys[0],), zs + (zs[0],), color=box_color, linewidth=1.5)
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
ax3d.set_title('3D View')

# Set equal scaling for 3D view
ax3d.set_box_aspect([1, 1, 1])

# Top-down (XY) view
ax_xy = fig.add_subplot(122)
ax_xy.scatter(points[:, 0], points[:, 1], s=0.2, c=points[:, 2], cmap='viridis', alpha=0.5)
for i, box in enumerate(boxes):
    corners = box.corners().T
    # Project to XY
    xy = corners[:, :2]
    xy = np.vstack([xy, xy[0]])  # close the box
    ax_xy.plot(xy[:, 0], xy[:, 1], color=colors(i)[:3], linewidth=2)
ax_xy.set_xlabel('X')
ax_xy.set_ylabel('Y')
ax_xy.set_title('Top-down (XY) View')
ax_xy.axis('equal')

# Set equal limits for both views to ensure same scale
x_min, x_max = ax_xy.get_xlim()
y_min, y_max = ax_xy.get_ylim()
z_min, z_max = ax3d.get_zlim()

# Use the same range for all axes
all_min = min(x_min, y_min, z_min)
all_max = max(x_max, y_max, z_max)

ax_xy.set_xlim(all_min, all_max)
ax_xy.set_ylim(all_min, all_max)
ax3d.set_xlim(all_min, all_max)
ax3d.set_ylim(all_min, all_max)
ax3d.set_zlim(all_min, all_max)

plt.tight_layout()
plt.show(block=True) 
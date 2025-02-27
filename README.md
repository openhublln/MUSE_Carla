# CARLA Multi-Sensor Data Collection Pipeline

A comprehensive pipeline for generating and visualizing synthetic data from the CARLA simulator. This tool enables collecting synchronized data from multiple sensors (cameras, LiDAR, semantic LiDAR, radar) in simulated scenes and provides interactive visualization tools for data analysis. Perfect for researchers and developers testing perception algorithms and sensor fusion in a controlled, repeatable environment.

## Requirements
- CARLA 0.10.0
- Python 3.7+
- Required Python packages:
  - numpy<2.0,>=1.24.4
  - opencv-python
  - open3d (Python ≤ 3.11)
  - Pillow
  - matplotlib
  - pygame
  - PyYAML

## Features
- Multi-sensor synchronized data collection:
  - RGB Cameras with automatic bounding box generation
  - LiDAR point clouds
  - Semantic LiDAR with CityScapes color mapping
  - Radar with intensity simulation
- Vehicle detection for multiple types:
  - Cars
  - Trucks
  - Buses
  - Motorcycles
- Configurable simulation parameters:
  - Weather conditions (preset or custom)
  - Traffic density
  - Scene duration
  - Sensor positioning
- Interactive visualization:
  - Multi-sensor synchronized playback
  - Grid-based display
  - Real-time data inspection
  - Aspect ratio preservation

## Pipeline Architecture
### 1. Configuration (config.yml)
- Centralized configuration for all simulation parameters
- Sensor setup (type, position, attributes)
- Scene parameters (duration, count)
- Weather and traffic settings
- Easy to modify without changing code

### 2. Data Collection (multi_sensor_collection.py)
- CARLA server connection and synchronous mode setup
- Scene and sensor folder creation
- Vehicle spawning and sensor attachment
- Synchronized data collection
- Data formats:
  - Cameras: PNG
  - LiDAR: NPY (x, y, z, intensity)
  - Semantic LiDAR: PLY + NPY (with semantic tags)
  - Radar: NPY (depth, altitude, azimuth, velocity, intensity)
  - Bounding Boxes: JSON

### 3. Data Visualization (multi_sensor_replay.py)
- Configuration-based sensor loading
- Common timestamp synchronization
- Grid-based visualization
- Interactive controls:
  - Space: Play/Pause
  - Left/Right arrows: Frame navigation
  - ESC: Exit

## Data Organization
```
_out/
  └── scene_X/
      ├── camera_1/
      ├── instance_camera_1/
      ├── lidar/
      ├── semantic_lidar/
      ├── radar/
      └── ...
```


## Usage Examples
Basic data collection:
```
python multi_sensor_collection.py
```

Visualize specific scene:
```
python multi_sensor_replay.py scene_2
```

Modify sensor configuration: Edit config.yml to add/remove sensors or change their parameters.


## License
This project is licensed under the terms of the MIT license.

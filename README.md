# CARLA Multi-Sensor Data Collection Pipeline

The CARLA Multi-Sensor Data Collection Pipeline is an end-to-end framework designed to simulate, capture, and analyze multi-modal sensor data in realistic urban environments using the CARLA simulator. This project provides a modular and highly configurable solution for autonomous vehicle research and development.

## Overview

The pipeline integrates several key components:

- **Flexible Configuration Management:**  
  Define simulation parameters and sensor specifications via a centralized YAML file or interactively through a graphical configuration editor.
- **Synchronized Data Acquisition:**  
  Attach a variety of sensors (RGB cameras, LiDAR, radar, GNSS, IMU) to an ego vehicle in CARLA. A tick-based synchronization mechanism ensures that all sensor data are temporally aligned, facilitating sensor fusion and perception tasks.
- **Automated Annotation Generation:**  
  Automatically generates bounding box annotations from instance segmentation images, streamlining the transition from raw data to training-ready datasets.
- **Interactive Replay and Visualization:**  
  A replay tool allows users to inspect multi-modal sensor data in a grid layout. Overlaid annotations and sensor metadata provide an interactive way to validate and refine simulation parameters.

## Architecture

The pipeline is divided into several main modules:
1. **Configuration Management:**  
   - **YAML Configuration File (`config.yml`):**  
     Centralized source for simulation and sensor settings.
   - **Graphical Configuration Editor (`config_editor.py`):**  
     A PyQt6-based tool for dynamically editing and previewing configuration parameters.
2. **Data Collection and Processing:**  
   - **Data Collection (`multi_sensor_collection.py`):**  
     Initializes simulation scenes, sets up sensor folders, attaches sensors with asynchronous callbacks, and ensures temporal synchronization.
   - **Annotation Generation (`generate_bbox_annotations.py`):**  
     Converts instance segmentation images to bounding box annotations stored in JSON format.
3. **Replay Tool:**  
   - **Multi-Sensor Replay (`multi_sensor_replay.py`):**  
     Loads synchronized sensor data and displays them in a grid layout with interactive controls for frame-by-frame analysis.

## Features

- **Dual-Mode Configuration:**  
  Supports both text-based YAML configuration and a user-friendly graphical editor for easy parameter tuning.
- **Sensor Diversity:**  
  Supports cameras, LiDAR, radar, GNSS, and IMU sensors with customizable attributes and transformation settings.
- **Temporal Synchronization:**  
  Ensures that sensor data across modalities are aligned by using a tick-based mechanism synchronized with CARLA’s physics engine.
- **Automated Post-Processing:**  
  Provides built-in routines for generating training-ready bounding box annotations from segmentation images.
- **Visualization and Replay:**  
  Offers an interactive replay tool that overlays annotations, timestamps, and sensor names to facilitate data validation and debugging.

## Sensor Catalogue

| Sensor Type                         | Attributes                                                                                                                                                                                                                                                                                      | Output Format                                                                                                                                                                             | Data Structure                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|-------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Camera (RGB, Semantic & Instance)** | image_size_x, image_size_y, fov, position, rotation.                                                                                                                                                                                                                                              | Images: `<timestamp>.png`<br>Annotation (if enabled): `<timestamp>_bbox.json`                                                                                                             | **PNG Image File:**<br>• RGB Cameras: A standard image saved as PNG.<br>• Semantic Segmentation: A PNG image where each pixel is remapped to a semantic palette (e.g., CityScapes).<br>• Instance Segmentation: A PNG image where the R channel holds the semantic class ID and the G & B channels encode unique instance IDs.<br><br>**JSON Annotation File:**<br>• image_file: Name of the corresponding PNG image.<br>• timestamp: Sensor timestamp.<br>• camera_data: An object with parameters (e.g., width, height, fov).<br>• bounding_boxes: A list of objects, each with a vehicle_id and bbox (list of floats in the order [x, y, width, height]). |
| **Radar**                           | horizontal_fov, vertical_fov, points_per_second, range, position, rotation.                                                                                                                                                                                                                       | `<timestamp>.npy`                                                                                                                                                                         | A NumPy array where each row corresponds to one radar detection. Each row contains 5 elements in the following order:<br>1. Depth (float): Distance to the object.<br>2. Elevation Angle (degrees).<br>3. Azimuth Angle (degrees).<br>4. Velocity (float): Relative speed.<br>5. Intensity (float): Signal strength calculated based on the depth with added noise.                                                                                                                                                                                                 |
| **LiDAR**                           | channels, range, points_per_second, rotation_frequency, upper_fov, lower_fov, horizontal_fov, atmosphere_attenuation_rate, dropoff_general_rate, dropoff_intensity_limit, dropoff_zero_intensity, noise_stddev, position, rotation.                                                           | `<timestamp>.npy`                                                                                                                                                                         | A NumPy array containing the raw data. The one-dimensional array of 32-bit floats is reshaped into an array with 4 columns representing:<br>• x, y, z (float32): Spatial coordinates.<br>• Intensity (float32): Reflectivity/attenuation value.                                                                                                                                                                                                                                                                                                              |
| **Semantic LiDAR**                  | channels, range, points_per_second, rotation_frequency, upper_fov, lower_fov, horizontal_fov, position, rotation.                                                                                                                                                                               | PLY file: `<timestamp>.ply`<br>NPY file: `<timestamp>.npy`                                                                                                                               | **PLY file:** A point cloud for 3D visualization.<br>**NPY file:** A structured NumPy array with the following fields:<br>• x, y, z (float32): 3D coordinates.<br>• cos_inc_angle (float32): Cosine of the incidence angle.<br>• object_idx (uint32): Unique object identifier.<br>• semantic_tag (uint32): Semantic class label.                                                                                                                                                                                                                                  |
| **GNSS**                            | noise_alt_bias, noise_alt_stddev, noise_lat_bias, noise_lat_stddev, noise_lon_bias, noise_lon_stddev, position, rotation.                                                                                                                                                                         | `<timestamp>.json`                                                                                                                                                                        | A JSON object containing:<br>• timestamp (in milliseconds).<br>• latitude (double).<br>• longitude (double).<br>• altitude (double).                                                                                                                                                                                                                                                                                                                                                                        |
| **IMU**                             | noise_accel_stddev_x, noise_accel_stddev_y, noise_accel_stddev_z, noise_gyro_stddev_x, noise_gyro_stddev_y, noise_gyro_stddev_z, noise_gyro_bias_x, noise_gyro_bias_y, noise_gyro_bias_z, position, rotation.                                                                          | `<timestamp>.json`                                                                                                                                                                        | A JSON object containing:<br>• timestamp (in milliseconds).<br>• accelerometer (float): An object with keys x, y, and z (m/s²).<br>• gyroscope (float): An object with keys x, y, and z (rad/s).<br>• compass (float): Compass reading (rad). |


## Installation

### System Requirements

- **CARLA Simulator:** v0.10.0  
- **Python:** 3.7+
- **Python Dependencies**
    - `numpy` (>=1.24.4, <2.0)
    - `opencv-python`
    - `PyYAML`
    - `pygame`
    - `matplotlib`
    - `PyQt6`
    - `Pillow`
    - `open3d` (for Python ≤ 3.11)

### Setup Steps

1. **Install CARLA Simulator:**  
   Follow the [quick start guide](https://carla-ue5.readthedocs.io/en/latest/start_quickstart/) for CARLA v0.10.0.
2. **Clone the Repository:**  
   Create a folder named `muse` inside the `Carla-0.10.0-Win64-Shipping/PythonAPI` directory and clone this project from GitHub.
3. **Setup Python Environment:**  
   Create and activate a Python virtual environment, then install the required packages using `pip`.
4. **Run the Configuration Editor:**  
   Execute `config_editor.py` to interactively set your simulation and sensor parameters.

## Usage

1. **Configuration:**  
   Edit simulation parameters, sensor specifications, and traffic settings either directly in `config.yml` or via the graphical editor. Changes are reflected in a live YAML preview.
2. **Simulation and Data Collection:**  
   - Launch the CARLA simulation.
   - The data collection module (`multi_sensor_collection.py`) initializes the scene, attaches sensors, and synchronizes data capture.
   - Sensor data is saved in a structured directory layout for each simulation scene.
3. **Automated Annotation:**  
   After simulation, bounding box annotations are automatically generated from instance segmentation images using the `generate_bbox_annotations.py` module.
4. **Replay and Verification:**  
   Use the replay tool (`multi_sensor_replay.py`) to inspect multi-modal sensor data. Navigate through frames, view overlaid annotations, and verify synchronization across sensors.

## Limitations and Future Improvements

- **Bounding Box Annotation:**  
  The current annotation process requires capturing two images per camera, which may affect performance when multiple sensors are recording simultaneously.
- **Ego Vehicle Handling:**  
  Automatic bounding box generation for the ego vehicle can degrade training data quality. A fix is planned for future updates.
- **Fixed Frame Rate:**  
  All sensors operate at a fixed frame rate (20Hz). Future releases aim to support individual sensor frame rate configuration.
- **Output Standardization:**  
  The output format does not yet conform to standards like NuScenes or KITTI. Future work will focus on standardization for improved interoperability.
- **Environmental Constraints:**  
  Currently, the simulation supports a single map and lacks weather variations (as per CARLA v0.10.0 limitations). Future enhancements may include additional maps and dynamic weather simulation.

## License

This project is licensed under the terms of the MIT license.

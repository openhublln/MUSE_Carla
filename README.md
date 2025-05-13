# CARLA Multi-Sensor Data Collection Pipeline

The CARLA Multi-Sensor Data Collection Pipeline is an end-to-end framework designed to simulate, capture, and analyze multi-modal sensor data in realistic urban environments using the CARLA simulator. This project provides a modular and highly configurable solution for autonomous vehicle research and development.

## Overview

The pipeline integrates several key components:

- **Flexible Configuration Management:**  
  Define simulation parameters and sensor specifications via a centralized YAML file or interactively through a graphical configuration editor.
- **Synchronized Data Acquisition:**  
  Attach a variety of sensors (RGB cameras, LiDAR, radar, GNSS, IMU) to an ego vehicle in CARLA. A tick-based synchronization mechanism ensures that all sensor data are temporally aligned, facilitating sensor fusion and perception tasks.
- **Automated Annotation Generation:**  
  Automatically generates bounding box annotations from instance segmentation images, streamlining the transition from raw data to training-ready datasets. Automatically generates projected 3D bounding boxes for vehicles from RGB camera data, including visibility estimation and robust handling of truncated objects using the Liang-Barsky algorithm
- **NuScenes Data Formatting:**
  A dedicated conversion module transforms the collected and annotated data into the widely-used NuScenes format, including keyframe selection, attribute inference, and points-in-box calculations.
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
   - **Data Collection (`multi_sensor_collection.py`, `sensor_processing.py`, `simulation_logic.py`):**  
     Initializes simulation scenes, sets up sensor folders, attaches sensors with asynchronous callbacks, ensures temporal synchronization, and saves ego vehicle pose.
   - **3D Bounding Box Export (`bounding_box_export.py`):**
     Projects 3D vehicle bounding boxes onto camera planes, clips them, calculates visibility, and saves detailed 3D annotation data.
   - **2D Annotation Generation (`generate_bbox_annotations.py`):**
     Converts instance segmentation images to 2D bounding box annotations stored in JSON format.
3.  **Data Formatting:**
   - **NuScenes Converter (`carla_to_nuscene_converter.py`):**
     Processes all collected data and annotations to produce a dataset compliant with the NuScenes format.
4. **Replay Tool:**  
   - **Multi-Sensor Replay (`multi_sensor_replay.py`, `replay_processing.py`):**  
     Loads synchronized sensor data and displays them in a grid layout with interactive controls for frame-by-frame analysis, supporting visualization of both 2D and projected 3D bounding boxes.

## Features

- **Dual-Mode Configuration:**  
  Supports both text-based YAML configuration and a user-friendly graphical editor for easy parameter tuning.
- **Sensor Diversity:**  
  Supports cameras, LiDAR, radar, GNSS, and IMU sensors with customizable attributes and transformation settings.
- **Temporal Synchronization:**  
  Ensures that sensor data across modalities are aligned by using a tick-based mechanism synchronized with CARLA’s physics engine.
- **Automated 3D & 2D Annotations:**
  Generates projected 3D bounding boxes with visibility for vehicles from RGB camera data. Optionally generates 2D bounding boxes from instance segmentation images.
- **NuScenes Export:**
  Provides tools to convert the collected dataset into the NuScenes format for standardization and broader compatibility.
- **Enhanced Visualization and Replay:**
  Offers an interactive replay tool that overlays 2D/3D annotations, timestamps, sensor names, and visibility information.

## Sensor Catalogue

| Sensor Type                         | Output Format                                                                                                                                                                             | Data Structure                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|-------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Camera (RGB, Semantic & Instance)** | Images: `<timestamp>.png`<br>2D Annotation (if enabled): `<timestamp>_bbox.json`<br>**3D Annotation (RGB only): `<timestamp>_3dbbox.json`**                                                                                                                              | **PNG Image File:**<br>• RGB Cameras: A standard image saved as PNG.<br>• Semantic Segmentation: A PNG image remapped to a semantic palette.<br>• Instance Segmentation: A PNG image (R: class ID, G&B: instance ID).<br><br>**2D JSON Annotation File:**<br>• `image_file`, `timestamp`, `camera_data`, `bounding_boxes` ([x, y, width, height]).<br><br>**3D JSON Annotation File:**<br>• For each detected vehicle: `actor_id`, `type`, `clipped_segments` (2D projected edges), `bbox_from_clipped` (2D extent), `velocity`, `pose` (3D world), `size` (3D NuScenes format: W,L,H), `visibility` (%). |
| **Radar**                           | `<timestamp>.npy`                                                                                                                                                                         | A NumPy array where each row corresponds to one radar detection. Each row contains 5 elements in the following order:<br>1. Depth (float): Distance to the object.<br>2. Elevation Angle (degrees).<br>3. Azimuth Angle (degrees).<br>4. Velocity (float): Relative speed.<br>5. Intensity (float): Signal strength calculated based on the depth with added noise.                                                                                                                                                                                                 |
| **LiDAR**                           | `<timestamp>.npy`                                                                                                                                                                         | A NumPy array containing the raw data. The one-dimensional array of 32-bit floats is reshaped into an array with 4 columns representing:<br>• x, y, z (float32): Spatial coordinates.<br>• Intensity (float32): Reflectivity/attenuation value.                                                                                                                                                                                                                                                                                                              |
| **Semantic LiDAR**                  | PLY file: `<timestamp>.ply`<br>NPY file: `<timestamp>.npy`                                                                                                                               | **PLY file:** A point cloud for 3D visualization.<br>**NPY file:** A structured NumPy array with the following fields:<br>• x, y, z (float32): 3D coordinates.<br>• cos_inc_angle (float32): Cosine of the incidence angle.<br>• object_idx (uint32): Unique object identifier.<br>• semantic_tag (uint32): Semantic class label.                                                                                                                                                                                                                                  |
| **GNSS**                            | `<timestamp>.json`                                                                                                                                                                        | A JSON object containing:<br>• timestamp (in milliseconds).<br>• latitude (double).<br>• longitude (double).<br>• altitude (double).                                                                                                                                                                                                                                                                                                                                                                        |
| **IMU**                             | `<timestamp>.json`                                                                                                                                                                        | A JSON object containing:<br>• timestamp (in milliseconds).<br>• accelerometer (float): An object with keys x, y, and z (m/s²).<br>• gyroscope (float): An object with keys x, y, and z (rad/s).<br>• compass (float): Compass reading (rad). |

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
    - `shapely`
    - `scipy`
    - `uuid`

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
   - The data collection module (`multi_sensor_collection.py`) initializes the scene, attaches sensors, synchronizes data capture, and generates 3D bounding box annotations.
   - Sensor data is saved in a structured directory layout for each simulation scene.
3. **Automated Annotation:**  
   After simulation, bounding box annotations are automatically generated from instance segmentation images using the `generate_bbox_annotations.py` module.
4. **NuScenes Conversion:**
   - Create a `converter_config.yml` specifying input/output paths and mappings.
   - Run the script (`carla_to_nuscene_converter.py`) to convert the collected data into the NuScenes format.  
4. **Replay and Verification:**  
   Use the replay tool (`multi_sensor_replay.py`) to inspect multi-modal sensor data. Navigate through frames, view overlaid annotations, and verify synchronization across sensors.

## Limitations and Future Improvements

- **3D Bounding Box Detection:**
  Currently focuses on a generic 'vehicle' category. Expansion to other vehicle types and pedestrians is planned. Attribute inference (e.g., 'parked' status) is basic and can be refined. Extensive validation of visibility and points-in-box algorithms across diverse scenarios is ongoing.
- **NuScenes Data Format:**
  Advanced features like `lidarseg.json` and `map.json` are not yet implemented. Camera intrinsic parameters need to be fully populated in `calibrated_sensor.json`.
- **Performance:**
  Capturing multiple images per camera (RGB, instance for 2D BBox) can impact performance. The new 3D BBox method only requires RGB.
- **Fixed Frame Rate:**
  All sensors operate at a fixed frame rate (20Hz). Future releases aim to support individual sensor frame rate configuration.
-  **Environmental Constraints:**
  Currently, the simulation supports a single map and lacks weather variations (as per CARLA v0.10.0 limitations).

## License

This project is licensed under the terms of the MIT license.

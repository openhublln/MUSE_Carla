# CARLA Multi-Sensor Data Collection Pipeline

End-to-end pipeline for collecting multi-modal sensor data in [CARLA 0.10.0](https://github.com/carla-simulator/carla/releases/tag/0.10.0) and converting it to NuScenes format. Supports RGB cameras, LiDAR, semantic LiDAR, radar, GNSS, and IMU — all output is structured to align with the NuScenes schema.

## Architecture

The pipeline has three independent stages: **collection**, **replay**, and **conversion**. Two config files drive the whole pipeline — `config.yml` for collection and `converter_config.yml` for NuScenes conversion.

### Stage 1 — Data Collection

`muse.py` (GUI) or `collection/multi_sensor_collection.py` (headless) is the entry point.

| File | Role |
|---|---|
| `collection/multi_sensor_collection.py` | Main loop: connects to CARLA, ticks all sensors in sync, writes raw data to `_out/` |
| `collection/traffic_setup.py` | Spawns the ego vehicle and NPC traffic (vehicles + pedestrians) |
| `collection/sensor_processing.py` | Attaches sensors to the ego vehicle; auto-injects a paired instance camera for each RGB camera that has `collect_bbox: true` |
| `collection/bounding_box_export.py` | Runs every tick: projects 3D bounding boxes of nearby actors onto each camera image and writes `*_3dbbox.json` |
| `collection/generate_bbox_annotations.py` | Runs once after all scenes: generates 2D bounding boxes from instance segmentation images and writes `*_bbox.json`. **Opt-in** — only active for cameras with `collect_bbox: true` |
| `collection/simulation_logic.py` | Creates the output folder structure (`scene_N/`) before collection starts |

### Stage 2 — Replay

`replay/multi_sensor_replay.py` loads a collected scene and plays it back frame by frame in a pygame window, with optional 2D or 3D bounding box overlay.

### Stage 3 — NuScenes Conversion

`conversion/carla_to_nuscene_converter.py` reads `_out/` and writes a NuScenes-compatible dataset to `nuscenes_format/`.

| File | Role |
|---|---|
| `sensor_calibrated_generators.py` | Writes `sensor.json` and `calibrated_sensor.json` |
| `log_generator.py` | Writes `log.json` from `_out/log_info.json` |
| `metadata_generators.py` | Writes `category.json`, `attribute.json`, `visibility.json` |
| `instance_generator.py` | Writes `instance.json` — one entry per unique actor across all scenes |
| `sample_generator.py` | Selects keyframes at the configured rate and writes `sample.json` |
| `sample_data_generator.py` | Converts raw files (PNG→JPEG, NPY→BIN/PCD) and writes `sample_data.json` |
| `annotation_generator.py` | Reads `*_3dbbox.json`, counts LiDAR/radar points per box, writes `sample_annotation.json` |
| `nuscenes_fixes.py` | Post-processing pass: fixes camera intrinsics, LiDAR quality flags, map paths |

### Key behaviours

- All sensors tick together in **synchronous CARLA mode** at `simulation.frequency_hz` (default **2 Hz**).
- **3D bounding boxes** are written automatically on every tick for every RGB camera — no configuration needed.
- **2D bounding boxes** are opt-in: set `collect_bbox: true` on an RGB camera in `config.yml`. This auto-spawns a paired instance segmentation camera and runs 2D bbox extraction after collection.
- `semantic_lidar` is collected but skipped during NuScenes conversion (commented out in `converter_config.yml`).

---

## Installation

**1. Download CARLA 0.10.0**

From the [CARLA releases page](https://github.com/carla-simulator/carla/releases/tag/0.10.0), download and extract the package for your OS (`Carla-0.10.0-Linux-Shipping` or `Carla-0.10.0-Win64-Shipping`).

**2. Clone this repository**

Create a `muse` folder inside `PythonAPI` and clone there:

```bash
cd Carla-0.10.0-Linux-Shipping/PythonAPI   # or Win64-Shipping
mkdir muse && cd muse
git clone <repo-url> MUSE_Carla
```

**3. Create a virtual environment and install dependencies**

```bash
cd MUSE_Carla
python -m venv carlavenv
source carlavenv/bin/activate          # Windows: carlavenv\Scripts\activate

# Install the CARLA Python client (not on PyPI)
pip install ../../carla/dist/carla-*-cp3*-linux_x86_64.whl   # adjust filename for your Python/OS

# Install project dependencies
pip install -r requirements.txt
```

> **Linux only:** the GUI also requires `sudo apt-get install -y libxcb-cursor0`

**4. Launch the GUI**

All scripts must be run from the `MUSE_Carla/` directory (config files are opened by relative path):

```bash
cd MUSE_Carla
python muse.py
```

---

## Usage

The recommended workflow goes through the GUI in order:

**1. Launch CARLA**  
Use the "Launch CARLA" button in the GUI, or start the CARLA server manually. The server must be running on `localhost:2000` before any other step.

**2. Configure**  
Set simulation parameters (number of scenes, duration, frequency) and add/edit sensors in the *Simulation* and *Sensors* tabs. Save to `config.yml`. Alternatively edit `config.yml` directly.

**3. Run data collection**  
Click "Run Simulation" in the GUI (blocks until complete), or run headless:
```bash
python collection/multi_sensor_collection.py
```
Output is written to `./_out/scene_N/`.

**4. Replay / verify**  
Inspect collected data frame by frame, with 2D and 3D bbox overlays:
```bash
python replay/multi_sensor_replay.py [scene_name] [2d|3d]
```
Or use the "Visualize" button in the GUI.

**5. Convert to NuScenes format**  
```bash
python conversion/carla_to_nuscene_converter.py converter_config.yml
```
Output is written to `./nuscenes_format/`.

---

## License

MIT

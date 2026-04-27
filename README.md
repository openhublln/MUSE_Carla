# CARLA Multi-Sensor Data Collection Pipeline

End-to-end pipeline for collecting multi-modal sensor data in [CARLA 0.10.0](https://github.com/carla-simulator/carla/releases/tag/0.10.0) and converting it to NuScenes format. Supports RGB cameras, LiDAR, semantic LiDAR, radar, GNSS, and IMU — all output is structured to align with the NuScenes schema.

## Overview & Architecture

The pipeline is driven by a single `config.yml` (simulation parameters, sensor layout, traffic) and a `converter_config.yml` (NuScenes conversion settings). Everything is Python, no build step.

```
config.yml / config_editor.py (PyQt6 GUI)
        │
        ▼
collection/multi_sensor_collection.py          ← main entry point for data collection
    ├── collection/sensor_processing.py        ← sensor attachment, callbacks, instance camera injection
    ├── collection/traffic_setup.py            ← ego vehicle + NPC spawning
    ├── collection/bounding_box_export.py      ← 3D bbox projection + visibility (runs every tick, per camera)
    ├── collection/simulation_logic.py         ← scene folder creation (tick loop is inline in collection)
    └── collection/generate_bbox_annotations.py← 2D bbox from instance segmentation (opt-in, post-collection)

conversion/carla_to_nuscene_converter.py       ← NuScenes conversion entry point
    ├── conversion/sensor_calibrated_generators.py
    ├── conversion/log_generator.py
    ├── conversion/metadata_generators.py
    ├── conversion/instance_generator.py
    ├── conversion/sample_generator.py
    ├── conversion/sample_data_generator.py
    ├── conversion/annotation_generator.py
    └── conversion/nuscenes_fixes.py

replay/multi_sensor_replay.py              ← frame-by-frame visualisation with 2D/3D bbox overlay
```

**Key design points:**
- All sensors tick together in synchronous CARLA mode at `simulation.frequency_hz` (default **2 Hz**).
- 3D bounding boxes are generated automatically for every RGB camera on every tick.
- 2D bounding boxes are **opt-in**: set `collect_bbox: true` on a camera in `config.yml`. This auto-spawns a paired instance segmentation camera and runs `generate_bbox_annotations.py` after collection.
- The committed `config.yml` has no GNSS or IMU sensors — add them manually if needed.
- `semantic_lidar` is silently skipped during NuScenes conversion (commented out in `converter_config.yml`).

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
pip install ../../dist/carla-*-cp3*-linux_x86_64.whl   # adjust filename for your Python/OS

# Install project dependencies
pip install -r requirements.txt
```

> **Linux only:** the GUI also requires `sudo apt-get install -y libxcb-cursor0`

**4. Launch the Configuration Editor**

All scripts must be run from the `MUSE_Carla/` directory (config files are opened by relative path):

```bash
cd MUSE_Carla
python config_editor.py
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

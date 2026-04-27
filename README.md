# CARLA Multi-Sensor Data Collection Pipeline

End-to-end pipeline for collecting multi-modal sensor data in [CARLA 0.10.0](https://github.com/carla-simulator/carla/releases/tag/0.10.0) and converting it to NuScenes format. Supports RGB cameras, LiDAR, semantic LiDAR, radar, GNSS, and IMU — all output is structured to align with the NuScenes schema.

## Architecture

Two config files drive the pipeline: `config.yml` (collection) and `converter_config.yml` (NuScenes conversion).

```
muse.py (GUI) / config.yml
        │
        ▼
collection/multi_sensor_collection.py      ← main loop: connects to CARLA, ticks sensors, writes _out/
    ├── traffic_setup.py                   ← spawns ego vehicle and NPC traffic
    ├── sensor_processing.py               ← attaches sensors; auto-injects instance camera when collect_bbox: true
    ├── bounding_box_export.py             ← every tick: projects 3D bboxes onto each camera, writes *_3dbbox.json
    ├── simulation_logic.py                ← creates scene_N/ output folders before collection
    └── generate_bbox_annotations.py       ← post-collection: 2D bboxes from instance segmentation (opt-in)

replay/multi_sensor_replay.py              ← frame-by-frame playback with 2D/3D bbox overlay

conversion/carla_to_nuscene_converter.py   ← reads _out/, writes nuscenes_format/
    ├── sensor_calibrated_generators.py    ← sensor.json + calibrated_sensor.json
    ├── log_generator.py                   ← log.json
    ├── metadata_generators.py             ← category / attribute / visibility tables
    ├── instance_generator.py              ← one instance entry per unique actor
    ├── sample_generator.py                ← keyframe selection
    ├── sample_data_generator.py           ← file conversion (PNG→JPEG, NPY→BIN/PCD)
    ├── annotation_generator.py            ← 3D bbox annotations + LiDAR/radar point counts
    └── nuscenes_fixes.py                  ← post-processing: intrinsics, LiDAR flags, map paths
```

- All sensors tick together in synchronous mode at `simulation.frequency_hz` (default **2 Hz**).
- **3D bboxes** are written every tick automatically for all RGB cameras.
- **2D bboxes** are opt-in: set `collect_bbox: true` on an RGB camera in `config.yml`.
- `semantic_lidar` is collected but skipped during NuScenes conversion.

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

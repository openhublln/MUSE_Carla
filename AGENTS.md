# AGENTS.md — MUSE_Carla

## Repo Identity

CARLA 0.10.0 multi-sensor data collection + NuScenes conversion pipeline. Python-only, no build system.
Must live at `<CARLA_root>/PythonAPI/muse/MUSE_Carla/`. All scripts open `config.yml` with a bare relative path — **always run from `MUSE_Carla/`**.

---

## Setup

No `requirements.txt` or `pyproject.toml`. Install manually:

```bash
python -m venv carlavenv && source carlavenv/bin/activate

# carla is NOT on PyPI — install from the CARLA distribution
pip install <CARLA_root>/PythonAPI/carla/dist/carla-0.10.0-cp3x-cp3x-linux_x86_64.whl

pip install "numpy>=1.24.4,<2.0" opencv-python PyYAML pygame matplotlib PyQt6 Pillow open3d shapely scipy
```

`open3d` requires Python ≤ 3.11 in theory; 0.19.0 has a cp312 wheel and installs fine on 3.12.
`shapely` and `scipy` are required — omitting them causes `ModuleNotFoundError` at runtime.

---

## Running

| Command | Purpose |
|---|---|
| `python config_editor.py` | GUI entry point (PyQt6) |
| `python multi_sensor_collection.py` | Headless data collection |
| `python multi_sensor_replay.py [scene_name] [2d\|3d]` | Replay / visualize |
| `python generate_bbox_annotations.py` | Post-process 2D bbox JSON |
| `python carla_to_nuscene_converter.py converter_config.yml` | Convert `_out/` → `nuscenes_format/` |
| `python change_map.py --map <MapName>` | Switch CARLA map (omit `--map` to list available) |

CARLA server must be running on `localhost:2000` before collection.

**Linux — Launch CARLA:** `config_editor.py` "Launch CARLA" button tries GUI terminals (`tilix`, `gnome-terminal`, `xterm`, `konsole`, `xfce4-terminal`) and falls back to a detached background process if all fail. Output logged to `<CARLA_root>/carla_launch.log`. Monitor with `tail -f`.

---

## Architecture

- **`config.yml`** — single source of truth for collection. Loaded fresh at the top of every script.
- **`converter_config.yml`** — config for the NuScenes converter. Separate from `config.yml`.
- **`sensor_processing.py`** auto-injects a paired `instance_<name>` camera for every RGB camera (`blueprint: sensor.camera.rgb`) with `collect_bbox: true`. Only RGB cameras qualify — semantic cameras are not paired.
- **Synchronous CARLA mode** is mandatory. Tick rate is `simulation.frequency_hz` from `config.yml`. The committed `config.yml` sets this to **2 Hz** — not 20. All sensors tick together via `queue.Queue`.
- **`run_simulation()` in `simulation_logic.py` is not called by the pipeline** — only `create_scene_folders()` from that module is used. The tick loop is inline in `multi_sensor_collection.py`. `simulation_logic.py` also contains French comments.
- **Annotation** runs after all scenes complete, not interleaved.
- **Map mask generation** (`generate_map_mask.py`) runs as a subprocess in the `finally` block of `multi_sensor_collection.py` after all scenes, with a 10-minute timeout.
- **`clean_scene_data()`** removes files whose timestamps are absent from any sensor folder. Skips folders whose path contains `"instance"`.
- `sensor_processing.py` reads `config.yml` inside the sensor callback on **every image frame** — significant I/O overhead at high frequency.

### GUI support modules (imported by `config_editor.py`)
- **`sensor_tab.py`** — sensor presets with exact default transform values; Camera_BackRight/BackLeft use unusual yaw values (`-225`, `225`).
- **`sensor_widgets.py`** — `get_config()` lowercases and underscores the sensor type (e.g. `"Semantic LIDAR"` → `"semantic_lidar"`). `collect_bbox` is only emitted for `"Camera"` (RGB) type. Note: `noise_gyro_stddev_y` widget label reads `"Accel StdDev Y"` (copy-paste bug — it is a gyro field).
- **`simulation_tab.py`** — GUI defaults differ from `config.yml`: `num_vehicles=30`, `num_pedestrians=10`, `frequency_hz=20`, `num_scenes=1`, `seconds_per_scene=20`. Seed is never exposed in the GUI; always writes `seed: None`.

---

## Sensor Config Conventions

- `type` must be one of: `camera`, `semantic_lidar`, `lidar`, `radar`, `gnss`, `imu`
- All sensor attributes are **YAML strings** (e.g., `image_size_x: '800.0'`). CARLA's `set_attribute()` requires strings.
- Add `collect_bbox: true` to an RGB camera to auto-generate 2D + 3D bbox annotations.
- Per-sensor frame rate is not independently configurable — all sensors run at `simulation.frequency_hz`.

---

## Output Layout

```
_out/
├── log_info.json                         # written once; used by NuScenes log_generator
├── <MapName>_basemap.png                 # written by generate_map_mask.py
├── <MapName>.json                        # map metadata (origin, scale, etc.)
└── scene_<N>/
    ├── ego_pose/<timestamp_ms>.json      # per-tick ego pose
    └── <SensorName>/
        ├── <timestamp_ms>.png / .npy / .json / .ply
        ├── <timestamp_ms>_bbox.json      # 2D bboxes (from generate_bbox_annotations.py)
        └── <timestamp_ms>_3dbbox.json   # 3D bboxes (from bounding_box_export.py)
```

| Sensor | Format / Shape |
|---|---|
| RGB / Semantic / Instance camera | `.png` |
| LiDAR | `.npy` shape `(-1, 4)`: x, y, z, intensity |
| Semantic LiDAR | `.npy` structured dtype + `.ply` |
| Radar | `.npy` shape `(-1, 5)`: depth, elevation_deg, azimuth_deg, velocity, intensity |
| GNSS / IMU | `.json` |
| Ego pose | `.json` per tick in `ego_pose/` subfolder |

Instance segmentation decode: `semantic_id = R`, `instance_id = (G << 8) | B`.

---

## NuScenes Conversion Pipeline

**Entry point:** `python carla_to_nuscene_converter.py converter_config.yml` (run from `MUSE_Carla/`).
Reads `_out/`, writes to `nuscenes_format/`.

**Module dependency chain:**
```
carla_to_nuscene_converter.py
    ├── nuscene_utils.py               # pure coord-transform utils; requires scipy + carla
    ├── sensor_calibrated_generators.py  # writes sensor.json + calibrated_sensor.json early
    ├── log_generator.py               # reads _out/log_info.json; hardcodes date "2024-01-01"
    ├── metadata_generators.py         # category / attribute / visibility tables
    ├── instance_generator.py          # reads *_3dbbox.json; first-seen category wins per actor
    ├── sample_generator.py            # keyframe selection; rate from converter_config.yml (default 2)
    ├── sample_data_generator.py       # converts files; ThreadPoolExecutor workers from converter_config.yml
    ├── annotation_generator.py        # reads *_3dbbox.json + LiDAR/radar .npy
    └── nuscenes_fixes.py              # post-processing: intrinsics, LiDAR quality, map paths
```

**File conversions by `sample_data_generator.py`:**
- Camera: PNG → JPEG
- LiDAR: NPY → BIN float32, 5 cols (x,y,z,intensity,pad), Y-axis flipped
- Radar: NPY → binary PCD, 18 cols with PCD header
- GNSS/IMU: JSON copy

**`ego_pose` must be generated before `sample_data_entries`** — `convert_all()` does this correctly; don't reorder.

**`semantic_lidar` is silently skipped** during conversion (commented out in `converter_config.yml` sensor_mappings).

**Coordinate system:** CARLA (Y-right) → NuScenes (Y-left) via Y-negation throughout. A camera-specific quaternion conversion exists in `nuscene_utils.py` but is currently unused — all sensors go through `carla_rotation_to_nuscenes_quaternion()`.

**`metadata_generators.py` quirks:**
- Category description is hardcoded as `"A car"` for `vehicle.car`, empty string for all others (TODO in code).
- Visibility tokens are fixed strings `"1"`–`"4"` (not UUIDs), matching NuScenes v1.0-mini format.
- `nuscenes_fixes.py` hardcodes `version_dir = 'v1.0'`.

---

## Hardcoded Values (not in config)

- Ego vehicle blueprint: `vehicle.lincoln.mkz` (in `traffic_setup.py`)
- CARLA host/port: `localhost:2000`, traffic manager: `8000`
- Max vehicle distance for 3D bbox export: `50 m`
- Min pixel count for **2D** bbox detection: `50 px` (in `generate_bbox_annotations.py`)
- Min pixel count for **3D** bbox visibility: `3 px` (in `bounding_box_export.py` as `MIN_BOX_PX`)
- 2D bbox vehicle semantic tags: `14` (Car), `15` (Truck), `16` (Bus), `18` (Motorcycle) — bicycle and pedestrian excluded
- Traffic spawn weights: 70% car, 10% truck, 10% bus, 5% motorcycle, 5% bicycle
- Map mask: captured at 400 m altitude, `medium` resolution (2048px), `--target-resolution 0.1`
- `change_map.py` hardcodes `fixed_delta_seconds = 0.05` regardless of `config.yml`
- `log_generator.py` hardcodes `logfile="carla_simulation_log"`, `vehicle="carla_ego_vehicle"`, `date="2024-01-01"`
- `sample_data_generator.py` `max_workers` comes from `converter_config.yml → performance.max_workers` (defaults to `min(32, cpu_count*2)`); there is no hardcoded `5`

---

## `config_editor.py` — Linux Notes

- "Launch CARLA" is Linux/Windows compatible. On Linux, tries `tilix` → `gnome-terminal` → `xterm` → `konsole` → `xfce4-terminal` (each with 1 s poll check). Falls back to detached background process if all fail.
- `parents[3]` from the script path resolves to CARLA root (`<CARLA_root>/PythonAPI/muse/MUSE_Carla/config_editor.py` → 4 levels up).
- The "Run Simulation" button calls `process.communicate()` — **the UI blocks until the simulation subprocess exits**.
- Requires `libxcb-cursor0`: `sudo apt-get install -y libxcb-cursor0`.

---

## No Tests, No CI, No Linting

No test suite, no `.github/workflows/`, no lint/format config.

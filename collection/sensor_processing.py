import os
import json
import math
import numpy as np
import carla
from pathlib import Path
from PIL import Image as PilImage
import io

ROOT = Path(__file__).resolve().parent.parent  # MUSE_Carla/

from bounding_box_export import export_3d_bboxes, get_static_vehicle_env_objects

def calculate_radar_intensity(depth):
    """Calcule l'intensité du signal radar."""
    rcs = 10          # Section efficace radar moyenne (m²)
    noise_floor = 1e-9
    ref_distance = 10 # Distance de référence (m)
    intensity = (ref_distance / depth) ** 4 * rcs
    noise = np.random.normal(0, noise_floor)
    return max(intensity + noise, 0)

def process_sensor_config(sensors_config):
    """Process sensor configuration and automatically add instance segmentation cameras.

    An instance camera is injected only for RGB cameras that have collect_bbox: true.
    The injected camera inherits the same transform but uses blueprint
    sensor.camera.instance_segmentation and its name is prefixed with 'instance_'.
    """
    processed_config = []
    
    for sensor in sensors_config:
        processed_config.append(sensor)
        
        if (sensor.get("type") == "camera" and 
            sensor.get("blueprint") == "sensor.camera.rgb" and 
            sensor.get("collect_bbox", False)):
            
            instance_camera = sensor.copy()
            instance_camera["name"] = f"instance_{sensor['name']}"
            instance_camera["blueprint"] = "sensor.camera.instance_segmentation"
            instance_camera.pop("collect_bbox", None)
            processed_config.append(instance_camera)
    
    return processed_config


# CityScapes palette for semantic segmentation (CARLA tag → RGB).
# 23 official tags; unknown tags map to black.
_CITYSCAPES_PALETTE = {
    0:  (0,   0,   0),    # Unlabeled
    1:  (128, 64,  128),  # Road
    2:  (244, 35,  232),  # SideWalk
    3:  (70,  70,  70),   # Building
    4:  (102, 102, 156),  # Wall
    5:  (190, 153, 153),  # Fence
    6:  (153, 153, 153),  # Pole
    7:  (250, 170, 30),   # TrafficLight
    8:  (220, 220, 0),    # TrafficSign
    9:  (107, 142, 35),   # Vegetation
    10: (152, 251, 152),  # Terrain
    11: (70,  130, 180),  # Sky
    12: (220, 20,  60),   # Pedestrian
    13: (255, 0,   0),    # Rider
    14: (0,   0,   142),  # Car
    15: (0,   0,   70),   # Truck
    16: (0,   60,  100),  # Bus
    17: (0,   80,  100),  # Train
    18: (0,   0,   230),  # Motorcycle
    19: (119, 11,  32),   # Bicycle
    20: (110, 190, 160),  # Static
    21: (170, 120, 50),   # Dynamic
    22: (55,  90,  80),   # Other
    23: (45,  60,  150),  # Water
    24: (157, 234, 50),   # RoadLine
    25: (81,  0,   81),   # Ground
    26: (150, 100, 100),  # Bridge
    27: (230, 150, 140),  # RailTrack
    28: (180, 165, 180),  # GuardRail
}

def _apply_cityscapes_palette(raw_rgba: np.ndarray) -> np.ndarray:
    """Convert CARLA semantic image (BGRA, R=tag) to CityScapes RGB palette.

    raw_rgba shape: (H, W, 4), dtype uint8, channel order BGRA.
    CARLA encodes the semantic tag in the Red channel.
    Returns (H, W, 3) uint8 RGB.
    """
    tags = raw_rgba[:, :, 2]  # R channel in BGRA layout
    h, w = tags.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for tag, color in _CITYSCAPES_PALETTE.items():
        mask = tags == tag
        out[mask] = color
    return out


# ---------------------------------------------------------------------------
# Thin sensor callback — runs in CARLA's internal callback thread.
# Must return as fast as possible.  Copies raw bytes immediately (safe here),
# then enqueues a plain-Python work item for the executor.
# NEVER call sensor_data.save_to_disk() from worker threads — it is not
# thread-safe and causes a C++ crash in the CARLA server.
# ---------------------------------------------------------------------------
def sensor_callback(sensor_data, sensor_queue, sensor_name, save_path,
                    blueprint_id, actor_snapshot, ego_transform):
    """Lightweight callback: copy raw bytes then enqueue for the I/O pool."""
    try:
        timestamp = int(sensor_data.timestamp * 1e3)

        if isinstance(sensor_data, carla.Image):
            # Copy BGRA pixels into a numpy array NOW, on the CARLA thread.
            # After this function returns, CARLA may reuse the buffer.
            arr = np.frombuffer(sensor_data.raw_data, dtype=np.uint8).copy()
            arr = arr.reshape((sensor_data.height, sensor_data.width, 4))
            # Serialize sensor_transform to plain Python — carla C++ objects
            # must NOT be accessed from worker threads.
            st = sensor_data.transform
            sensor_tf = {
                'x': st.location.x, 'y': st.location.y, 'z': st.location.z,
                'pitch': st.rotation.pitch, 'yaw': st.rotation.yaw, 'roll': st.rotation.roll,
                'matrix': [list(row) for row in st.get_matrix()],
            }
            payload = ('image', arr, sensor_data.width, sensor_data.height,
                       sensor_data.fov, sensor_tf)

        elif isinstance(sensor_data, carla.SemanticLidarMeasurement):
            # Copy structured point cloud bytes.
            raw = bytes(sensor_data.raw_data)
            payload = ('semantic_lidar', raw)

        elif isinstance(sensor_data, carla.LidarMeasurement):
            raw = np.frombuffer(sensor_data.raw_data, dtype=np.float32).copy()
            payload = ('lidar', raw)

        elif isinstance(sensor_data, carla.RadarMeasurement):
            pts = []
            for detect in sensor_data:
                azi = math.degrees(detect.azimuth)
                alt = math.degrees(detect.altitude)
                intensity = calculate_radar_intensity(detect.depth)
                pts.append([detect.depth, alt, azi, detect.velocity, intensity])
            payload = ('radar', pts)

        elif isinstance(sensor_data, carla.IMUMeasurement):
            payload = ('imu', {
                "timestamp": timestamp,
                "accelerometer": {
                    "x": sensor_data.accelerometer.x,
                    "y": sensor_data.accelerometer.y,
                    "z": sensor_data.accelerometer.z,
                },
                "gyroscope": {
                    "x": sensor_data.gyroscope.x,
                    "y": sensor_data.gyroscope.y,
                    "z": sensor_data.gyroscope.z,
                },
                "compass": sensor_data.compass,
            })

        elif isinstance(sensor_data, carla.GnssMeasurement):
            payload = ('gnss', {
                "timestamp": timestamp,
                "latitude":  sensor_data.latitude,
                "longitude": sensor_data.longitude,
                "altitude":  sensor_data.altitude,
            })

        else:
            return  # unknown sensor type — drop

        sensor_queue.put((payload, timestamp, sensor_name, save_path,
                          blueprint_id, actor_snapshot, ego_transform))
    except Exception as e:
        print(f"Error queuing sensor data for {sensor_name}: {e}")


# ---------------------------------------------------------------------------
# Worker function — called by the ThreadPoolExecutor on a worker thread.
# Receives only plain Python / NumPy data — zero CARLA client calls here.
# ---------------------------------------------------------------------------
def write_sensor_data(payload_tuple, timestamp, sensor_name, save_path,
                      blueprint_id, actor_snapshot, ego_transform,
                      static_vehicles, done_queue):
    """Serialise one sensor frame to disk.  Runs on an executor worker thread.
    No CARLA client calls allowed here — not thread-safe."""
    try:
        sensor_folder = os.path.join(save_path, sensor_name)
        kind = payload_tuple[0]

        if kind == 'image':
            _, arr, width, height, fov, transform = payload_tuple
            img_path = os.path.join(sensor_folder, f"{timestamp}.png")

            if blueprint_id == "sensor.camera.semantic_segmentation":
                rgb = _apply_cityscapes_palette(arr)
                PilImage.fromarray(rgb, 'RGB').save(img_path)
            else:
                # BGRA → RGBA for Pillow, then save as PNG
                rgba = arr[:, :, [2, 1, 0, 3]]  # BGRA → RGBA
                PilImage.fromarray(rgba, 'RGBA').save(img_path)

            # 3D bbox: only for RGB cameras; pass pre-copied transform + fov.
            if (blueprint_id == "sensor.camera.rgb" and
                    actor_snapshot is not None and
                    ego_transform is not None):
                try:
                    export_3d_bboxes(
                        arr, width, height, fov, transform,
                        sensor_folder, timestamp,
                        actor_snapshot, ego_transform,
                        static_vehicles=static_vehicles
                    )
                except Exception as e:
                    print(f"Warning: 3D bbox export failed for {sensor_name}: {e}")

        elif kind == 'semantic_lidar':
            _, raw = payload_tuple
            points = np.frombuffer(raw, dtype=np.dtype([
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('cos_inc_angle', np.float32),
                ('object_idx', np.uint32), ('semantic_tag', np.uint32)
            ]))
            npy_path = os.path.join(sensor_folder, f"{timestamp}.npy")
            np.save(npy_path, points)
            # Write PLY manually — no save_to_disk
            ply_path = os.path.join(sensor_folder, f"{timestamp}.ply")
            _write_semantic_lidar_ply(points, ply_path)

        elif kind == 'lidar':
            _, raw = payload_tuple
            pts = raw.reshape((-1, 4))
            np.save(os.path.join(sensor_folder, f"{timestamp}.npy"), pts)

        elif kind == 'radar':
            _, pts = payload_tuple
            arr = np.array(pts, dtype=np.float32)
            np.save(os.path.join(sensor_folder, f"{timestamp}.npy"), arr)

        elif kind == 'imu':
            _, data = payload_tuple
            with open(os.path.join(sensor_folder, f"{timestamp}.json"), 'w') as f:
                json.dump(data, f, separators=(',', ':'))

        elif kind == 'gnss':
            _, data = payload_tuple
            with open(os.path.join(sensor_folder, f"{timestamp}.json"), 'w') as f:
                json.dump(data, f, separators=(',', ':'))

        done_queue.put((timestamp, sensor_name))

    except Exception as e:
        import traceback
        print(f"Error writing sensor data for {sensor_name}: {e}")
        traceback.print_exc()
        done_queue.put((0, sensor_name))


def _write_semantic_lidar_ply(points, path):
    """Write semantic LiDAR structured array to ASCII PLY without CARLA."""
    n = len(points)
    lines = [
        "ply", "format ascii 1.0",
        f"element vertex {n}",
        "property float x", "property float y", "property float z",
        "property float cos_inc_angle",
        "property uint object_idx", "property uint semantic_tag",
        "end_header",
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
        for p in points:
            f.write(f"{p['x']} {p['y']} {p['z']} "
                    f"{p['cos_inc_angle']} {p['object_idx']} {p['semantic_tag']}\n")


def clean_scene_data(scene_path, sensor_names):
    """
    Nettoie le jeu de données d'une scène en supprimant les fichiers dont le timestamp
    n'est pas présent dans tous les dossiers de capteurs.
    """
    ts_dict = {}
    for sensor in sensor_names:
        sensor_folder = os.path.join(scene_path, sensor)
        if not os.path.isdir(sensor_folder):
            continue

        # Skip instance segmentation folders entirely
        if "instance" in sensor_folder:
            continue

        files = []
        for f in os.listdir(sensor_folder):
            if any(f.endswith(ext) for ext in ['.png', '.npy', '.json', '.ply']):
                if "_3dbbox.json" in f:
                    continue
                if f.endswith('.ply'):
                    npy_file = f[:-4] + '.npy'
                    if not os.path.exists(os.path.join(sensor_folder, npy_file)):
                        continue
                files.append(f)

        ts_set = set(os.path.splitext(f)[0] for f in files)
        if ts_set:
            ts_dict[sensor] = ts_set

    if not ts_dict:
        print(f"Warning: No valid files found in {scene_path}")
        return

    common_ts = set.intersection(*ts_dict.values())

    if not common_ts:
        print(f"Warning: No common timestamps found in {scene_path}")
        return

    print(f"Found {len(common_ts)} common timestamps across all sensors")

    deleted_count = 0
    for sensor in sensor_names:
        sensor_folder = os.path.join(scene_path, sensor)
        if not os.path.isdir(sensor_folder):
            continue

        for file_name in os.listdir(sensor_folder):
            if "_3dbbox.json" in file_name:
                base_ts = file_name.split("_3dbbox.json")[0]
                if base_ts in common_ts:
                    continue

            base_name = os.path.splitext(file_name)[0]
            if base_name not in common_ts:
                file_path = os.path.join(sensor_folder, file_name)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} non-synchronized files")
    else:
        print("All files are properly synchronized")

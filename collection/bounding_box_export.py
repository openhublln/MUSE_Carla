import os
import json
import numpy as np
import carla
from shapely.geometry import Polygon, box

# --- Simple in-file selector for which categories to export ---
# Categories align with CARLA 0.10.0 UE5 catalogue mapped to NuScenes leaf categories.
EXPORT_BBOX3D_CATEGORIES = {
    "vehicle.car",
    "vehicle.truck",
    "vehicle.emergency.police",
    "vehicle.emergency.ambulance",
    "vehicle.bus.rigid",
    "human.pedestrian",
}

# Blueprint ID → NuScenes category (must match traffic_setup.py BLUEPRINT_TO_NUSCENES)
_BLUEPRINT_TO_NUSCENES = {
    'vehicle.ue4.audi.tt':        'vehicle.car',
    'vehicle.dodge.charger':      'vehicle.car',
    'vehicle.taxi.ford':          'vehicle.car',
    'vehicle.lincoln.mkz':        'vehicle.car',
    'vehicle.ue4.mercedes.ccc':   'vehicle.car',
    'vehicle.mini.cooper':        'vehicle.car',
    'vehicle.nissan.patrol':      'vehicle.car',
    'vehicle.sprinter.mercedes':  'vehicle.car',
    'vehicle.carlacola.actors':   'vehicle.truck',
    'vehicle.firetruck.actors':   'vehicle.truck',
    'vehicle.dodgecop.charger':   'vehicle.emergency.police',
    'vehicle.ambulance.ford':     'vehicle.emergency.ambulance',
    'vehicle.fuso.mitsubishi':    'vehicle.bus.rigid',
}

# Distance and size thresholds
MAX_DISTANCE_METERS = 50.0
MIN_BOX_PX = 3

# Offset added to carla.EnvironmentObject.id values so they never collide with
# dynamic actor IDs (which are small integers assigned per-episode by CARLA).
STATIC_VEHICLE_ID_OFFSET = 1_000_000

# --- Actor category classification ---
def classify_actor_category(type_id: str) -> str:
    """Classify a CARLA actor into a NuScenes leaf category string.

    Accepts a type_id string (not the actor object itself — avoids an RPC call).
    Returns one of: vehicle.car, vehicle.truck, vehicle.emergency.police,
                    vehicle.emergency.ambulance, vehicle.bus.rigid, human.pedestrian
    Returns None if the actor type is not in the supported set.
    """
    if not type_id:
        return None
    if type_id.startswith('vehicle.'):
        return _BLUEPRINT_TO_NUSCENES.get(type_id.lower(), 'vehicle.car')
    if type_id.startswith('walker.pedestrian'):
        return 'human.pedestrian'
    return None


def classify_static_vehicle_category(name: str) -> str:
    """Infer a NuScenes category for a static world vehicle from its UE asset name."""
    name_lower = name.lower()
    if 'truck' in name_lower or 'van' in name_lower:
        return 'vehicle.truck'
    if 'bus' in name_lower:
        return 'vehicle.bus.rigid'
    return 'vehicle.car'

# --- Liang-Barsky Line Clipping Algorithm ---
def liang_barsky_clip(x1, y1, x2, y2, x_min, y_min, x_max, y_max):
    """Clips a line segment to a rectangular window."""
    dx = x2 - x1
    dy = y2 - y1
    t0, t1 = 0.0, 1.0
    checks = (
        (-dx, x1 - x_min), (dx, x_max - x1),
        (-dy, y1 - y_min), (dy, y_max - y1)
    )
    for p, q in checks:
        if p == 0:
            if q < 0: return None
        else:
            r = q / p
            if p < 0:
                if r > t1: return None
                if r > t0: t0 = r
            else:
                if r < t0: return None
                if r < t1: t1 = r
    clipped_x1 = x1 + t0 * dx
    clipped_y1 = y1 + t0 * dy
    clipped_x2 = x1 + t1 * dx
    clipped_y2 = y1 + t1 * dy
    if abs(clipped_x1 - clipped_x2) < 1e-6 and abs(clipped_y1 - clipped_y2) < 1e-6:
        return None
    return ((clipped_x1, clipped_y1), (clipped_x2, clipped_y2))

# Define indices for the 12 edges of a cuboid
EDGES_INDICES = [
    [0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
]

def build_projection_matrix(w, h, fov):
    """ Builds the camera intrinsic matrix K. """
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    """ Calculates the 2D projection of a 3D world point. """
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera_homogeneous = np.dot(w2c, point)
    is_behind = point_camera_homogeneous[0] <= 1e-4
    if is_behind:
        point_camera_homogeneous[0] = 1e-3
    point_camera = np.array([point_camera_homogeneous[1],
                              -point_camera_homogeneous[2],
                              point_camera_homogeneous[0]])
    point_img = np.dot(K, point_camera)
    if abs(point_img[2]) < 1e-4:
        return np.array([-1e6, -1e6]), True
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2], is_behind


def get_image_point_xyz(xyz, K, w2c):
    """Like get_image_point but accepts a plain [x, y, z] array/list."""
    point = np.array([xyz[0], xyz[1], xyz[2], 1.0])
    p = np.dot(w2c, point)
    is_behind = p[0] <= 1e-4
    if is_behind:
        p[0] = 1e-3
    pc = np.array([p[1], -p[2], p[0]])
    pi = np.dot(K, pc)
    if abs(pi[2]) < 1e-4:
        return np.array([-1e6, -1e6]), True
    pi[0] /= pi[2]
    pi[1] /= pi[2]
    return pi[0:2], is_behind


def _clip_edges(projected_vertices, image_w, image_h):
    """Clip bounding box edges against the image plane. Returns (segments, all_pts)."""
    clipped_segments = []
    all_pts = []
    for i, j in EDGES_INDICES:
        p1, behind1 = projected_vertices[i]
        p2, behind2 = projected_vertices[j]
        if behind1 and behind2:
            continue
        clip = liang_barsky_clip(p1[0], p1[1], p2[0], p2[1], 0.0, 0.0, image_w, image_h)
        if clip is not None:
            seg = [list(clip[0]), list(clip[1])]
            clipped_segments.append(seg)
            all_pts.extend([seg[0], seg[1]])
    return clipped_segments, all_pts


def _bbox_from_pts(all_pts):
    """Return [x_min, y_min, w, h] or None if box is too small."""
    if not all_pts:
        return None
    pts_arr = np.array(all_pts)
    x_min, y_min = pts_arr.min(axis=0)
    x_max, y_max = pts_arr.max(axis=0)
    w = max(0.0, float(x_max - x_min))
    h = max(0.0, float(y_max - y_min))
    if w < MIN_BOX_PX or h < MIN_BOX_PX:
        return None
    return [float(x_min), float(y_min), w, h]

def compute_visibility(clipped_segments, bbox_from_clipped, image_width, image_height):
    """
    Compute the visibility percentage of a bounding box.

    Args:
        clipped_segments: List of clipped line segments [[[x1,y1], [x2,y2]], ...]
        bbox_from_clipped: [x_min, y_min, width, height] of the bounding box
        image_width: Width of the image
        image_height: Height of the image

    Returns:
        float: Visibility percentage (0-100)
    """
    if not clipped_segments or not bbox_from_clipped:
        return 0.0

    x_min, y_min, width, height = bbox_from_clipped

    is_partially_outside = (
        x_min < 0 or y_min < 0 or
        x_min + width > image_width or
        y_min + height > image_height
    )

    try:
        points = []
        for segment in clipped_segments:
            points.extend(segment)

        if len(points) < 3:
            return 0.0

        points_array = np.array(points)
        start_idx = np.lexsort((points_array[:, 0], points_array[:, 1]))[0]
        start_point = points_array[start_idx]
        remaining_points = np.delete(points_array, start_idx, axis=0)
        angles = np.arctan2(remaining_points[:, 1] - start_point[1],
                            remaining_points[:, 0] - start_point[0])
        sorted_indices = np.argsort(angles)
        sorted_points = remaining_points[sorted_indices]
        ordered_points = np.vstack((start_point, sorted_points))

        visible_polygon = Polygon(ordered_points)
        if not visible_polygon.is_valid:
            return 0.0

        visible_area = visible_polygon.area

        if is_partially_outside:
            bbox_polygon = box(x_min, y_min, x_min + width, y_min + height)
            image_polygon = box(0, 0, image_width, image_height)
            total_area = bbox_polygon.intersection(image_polygon).area
        else:
            total_area = width * height

        if total_area <= 0:
            return 0.0

        visibility = max(0.0, min(100.0, (visible_area / total_area) * 100))
        return visibility

    except Exception:
        return 0.0


_STATIC_VEHICLE_LABELS = {
    carla.CityObjectLabel.Car,
    carla.CityObjectLabel.Truck,
    carla.CityObjectLabel.Bus,
    carla.CityObjectLabel.Motorcycle,
    carla.CityObjectLabel.Bicycle,
}

def get_static_vehicle_env_objects(world):
    """Query static world vehicles once at startup. Returns plain Python dicts.

    Serializes all carla C++ objects to plain Python so the result is safe
    to pass to worker threads without any further CARLA client access.
    """
    try:
        raw = [
            obj for obj in world.get_environment_objects(carla.CityObjectLabel.Any)
            if obj.type in _STATIC_VEHICLE_LABELS
        ]
    except Exception as e:
        print(f"Warning: Could not retrieve static vehicle environment objects: {e}")
        return []

    # Pass 1: deduplicate by name — same asset name = same physical vehicle,
    # keep only the sub-mesh with the largest bounding-box volume.
    by_name = {}
    for obj in raw:
        name = obj.name
        ext  = obj.bounding_box.extent
        vol  = ext.x * ext.y * ext.z
        if name not in by_name or vol > by_name[name][0]:
            by_name[name] = (vol, obj)
    after_name_dedup = [obj for _, obj in by_name.values()]

    # Pass 2: spatial dedup — objects with different names but bounding-box
    # origins within 0.5 m of each other are merged (largest volume wins).
    best = {}
    for obj in after_name_dedup:
        loc = obj.bounding_box.location
        key = (round(loc.x * 2) / 2, round(loc.y * 2) / 2, round(loc.z * 2) / 2)
        ext = obj.bounding_box.extent
        vol = ext.x * ext.y * ext.z
        if key not in best or vol > best[key][0]:
            best[key] = (vol, obj)

    deduped = []
    for _, obj in best.values():
        bb  = obj.bounding_box
        loc = bb.location
        ext = bb.extent
        # Compute world vertices once here on main thread, store as plain lists
        import carla as _carla
        try:
            verts = [[v.x, v.y, v.z]
                     for v in bb.get_world_vertices(_carla.Transform())]
        except Exception:
            verts = []
        deduped.append({
            'id':     obj.id,
            'name':   obj.name,
            'loc_x':  loc.x, 'loc_y': loc.y, 'loc_z': loc.z,
            'ext_x':  ext.x, 'ext_y': ext.y, 'ext_z': ext.z,
            'verts':  verts,
        })

    if len(deduped) < len(raw):
        print(f"Static vehicles: {len(raw)} env objects → {len(deduped)} after deduplication.")
    return deduped


def export_3d_bboxes(img_arr, image_w, image_h, fov, sensor_transform,
                     save_path, timestamp, actor_snapshot, ego_transform,
                     static_vehicles=None):
    """
    Export 3D bounding box edges projected and clipped onto the camera image.

    Parameters
    ----------
    img_arr        : np.ndarray (H, W, 4) BGRA uint8 — raw pixel data, already
                     copied on the CARLA callback thread.  Used only for timestamp
                     derivation here; not modified.
    image_w        : int   — camera width in pixels
    image_h        : int   — camera height in pixels
    fov            : float — camera horizontal field of view in degrees
    sensor_transform : carla.Transform — camera world transform (from snapshot)
    save_path      : str  — directory to write the _3dbbox.json file
    actor_snapshot : dict — pre-fetched actor data built once per tick by the
                            main thread (see build_actor_snapshot()).
                            Keys are actor IDs; values are dicts with keys:
                            'type_id', 'transform', 'bounding_box', 'velocity'.
    ego_transform  : carla.Transform — ego vehicle transform, pre-fetched per tick.
    static_vehicles: list of carla.EnvironmentObject — cached at startup.

    This function performs ZERO RPC calls to the CARLA server.
    This function contains ZERO carla C++ object access.
    All inputs are plain Python dicts, lists, floats, and numpy arrays.
    """
    image_w = float(image_w)
    image_h = float(image_h)
    fov     = float(fov)

    K = build_projection_matrix(image_w, image_h, fov)

    # Build w2c matrix from the serialized sensor transform dict
    w2c = np.array(sensor_transform['matrix'])
    # Invert to get world-to-camera
    w2c = np.linalg.inv(w2c)

    # Forward vector from sensor rotation (yaw/pitch in degrees)
    import math as _math
    yaw   = _math.radians(sensor_transform['yaw'])
    pitch = _math.radians(sensor_transform['pitch'])
    fwd_x = _math.cos(pitch) * _math.cos(yaw)
    fwd_y = _math.cos(pitch) * _math.sin(yaw)
    fwd_z = _math.sin(pitch)

    sx, sy, sz = sensor_transform['x'], sensor_transform['y'], sensor_transform['z']
    ex, ey, ez = ego_transform['x'],    ego_transform['y'],    ego_transform['z']

    output_data = []

    # ------------------------------------------------------------------
    # Dynamic actors — all data is plain Python dicts.
    # ------------------------------------------------------------------
    for actor_id, ainfo in (actor_snapshot or {}).items():
        category = classify_actor_category(ainfo['type_id'])
        if category is None or category not in EXPORT_BBOX3D_CATEGORIES:
            continue

        at = ainfo['transform']
        ax, ay, az = at['x'], at['y'], at['z']

        dist = _math.sqrt((ax-ex)**2 + (ay-ey)**2 + (az-ez)**2)
        if dist > MAX_DISTANCE_METERS:
            continue

        dot = fwd_x*(ax-sx) + fwd_y*(ay-sy) + fwd_z*(az-sz)
        if dot < 1:
            continue

        # Reconstruct world vertices from bounding box + actor transform matrix
        bb  = ainfo['bounding_box']
        ext_x, ext_y, ext_z = bb['ext_x'], bb['ext_y'], bb['ext_z']
        # Local bbox corners
        corners = np.array([
            [ ext_x,  ext_y, -ext_z], [ ext_x, -ext_y, -ext_z],
            [-ext_x,  ext_y, -ext_z], [-ext_x, -ext_y, -ext_z],
            [ ext_x,  ext_y,  ext_z], [ ext_x, -ext_y,  ext_z],
            [-ext_x,  ext_y,  ext_z], [-ext_x, -ext_y,  ext_z],
        ])
        # Offset by bbox local location
        corners += np.array([bb['loc_x'], bb['loc_y'], bb['loc_z']])
        # Rotate + translate to world space using actor transform matrix
        actor_mat = np.array(at['matrix'])
        ones = np.ones((8, 1))
        corners_h = np.hstack([corners, ones])          # (8, 4)
        world_verts = (actor_mat @ corners_h.T).T[:, :3]  # (8, 3)

        size = [ext_y * 2, ext_x * 2, ext_z * 2]

        projected_vertices = [get_image_point_xyz(v, K, w2c) for v in world_verts]

        clipped_segments, all_pts = _clip_edges(projected_vertices, image_w, image_h)
        if not clipped_segments:
            continue

        bbox_from_clipped = _bbox_from_pts(all_pts)
        if bbox_from_clipped is None:
            continue

        v = ainfo['velocity']
        vel_mag = _math.sqrt(v['x']**2 + v['y']**2 + v['z']**2)

        output_data.append({
            "actor_id": actor_id,
            "type": "vehicle" if category.startswith('vehicle.') else "pedestrian",
            "category": category,
            "clipped_segments": clipped_segments,
            "bbox_from_clipped": bbox_from_clipped,
            "velocity": {"x": v['x'], "y": v['y'], "z": v['z'], "magnitude": vel_mag},
            "pose": {
                "actor_id": actor_id,
                "timestamp": timestamp,
                "translation": {"x": ax, "y": ay, "z": az},
                "rotation": {"pitch": at['pitch'], "yaw": at['yaw'], "roll": at['roll']},
            },
            "size": size,
            "visibility": compute_visibility(clipped_segments, bbox_from_clipped,
                                             image_w, image_h),
        })

    # ------------------------------------------------------------------
    # Static vehicles — pre-serialized plain Python dicts.
    # ------------------------------------------------------------------
    for env_obj in (static_vehicles or []):
        try:
            ox, oy, oz = env_obj['loc_x'], env_obj['loc_y'], env_obj['loc_z']

            dist = _math.sqrt((ox-ex)**2 + (oy-ey)**2 + (oz-ez)**2)
            if dist > MAX_DISTANCE_METERS:
                continue

            dot = fwd_x*(ox-sx) + fwd_y*(oy-sy) + fwd_z*(oz-sz)
            if dot < 1:
                continue

            # Vertices already in world space (computed once at startup)
            world_verts = np.array(env_obj['verts'])
            if len(world_verts) == 0:
                continue

            ext_x = env_obj['ext_x']; ext_y = env_obj['ext_y']; ext_z = env_obj['ext_z']
            size = [ext_y * 2, ext_x * 2, ext_z * 2]

            projected_vertices = [get_image_point_xyz(v, K, w2c) for v in world_verts]

            clipped_segments, all_pts = _clip_edges(projected_vertices, image_w, image_h)
            if not clipped_segments:
                continue

            bbox_from_clipped = _bbox_from_pts(all_pts)
            if bbox_from_clipped is None:
                continue

            category = classify_static_vehicle_category(env_obj['name'])
            static_actor_id = env_obj['id'] + STATIC_VEHICLE_ID_OFFSET

            output_data.append({
                "actor_id":   static_actor_id,
                "type":       "static_vehicle",
                "category":   category,
                "is_static":  True,
                "clipped_segments":  clipped_segments,
                "bbox_from_clipped": bbox_from_clipped,
                "velocity": {"x": 0.0, "y": 0.0, "z": 0.0, "magnitude": 0.0},
                "pose": {
                    "actor_id":  static_actor_id,
                    "timestamp": timestamp,
                    "translation": {"x": ox, "y": oy, "z": oz},
                    "rotation":    {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
                "size":       size,
                "visibility": 100.0,
            })
        except Exception as e:
            print(f"Warning: Error processing static vehicle: {e}")
            continue

    # --- Save to JSON ---
    output_file = os.path.join(save_path, f"{timestamp}_3dbbox.json")
    try:
        os.makedirs(save_path, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, separators=(',', ':'))
    except Exception as e:
        print(f"Error writing JSON file {output_file}: {e}")

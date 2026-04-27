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

# --- Actor category classification ---
def classify_actor_category(actor: carla.Actor) -> str:
    """Classify a CARLA actor into a NuScenes leaf category string.

    Uses exact blueprint ID matching against the CARLA 0.10.0 UE5 catalogue.
    Returns one of: vehicle.car, vehicle.truck, vehicle.emergency.police,
                    vehicle.emergency.ambulance, vehicle.bus.rigid, human.pedestrian
    Returns None if the actor type is not in the supported set.
    """
    try:
        type_id = getattr(actor, 'type_id', '') or ''
    except Exception:
        type_id = ''

    if type_id.startswith('vehicle.'):
        # Exact match first; fall back to vehicle.car for any unknown blueprint
        return _BLUEPRINT_TO_NUSCENES.get(type_id.lower(), 'vehicle.car')

    if type_id.startswith('walker.pedestrian'):
        return 'human.pedestrian'

    return None

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
            else: # p > 0
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
        point_camera_homogeneous[0] = 1e-3 # Avoid div by zero/neg
    point_camera = np.array([point_camera_homogeneous[1], -point_camera_homogeneous[2], point_camera_homogeneous[0]])
    point_img = np.dot(K, point_camera)
    if abs(point_img[2]) < 1e-4:
         return np.array([-1e6, -1e6]), True
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2], is_behind

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
        print("No clipped segments or bbox provided")
        return 0.0
        
    # Get bounding box parameters
    x_min, y_min, width, height = bbox_from_clipped
    
    # Debug print
    print(f"\nDebug - Bounding Box:")
    print(f"Position: ({x_min}, {y_min})")
    print(f"Size: {width}x{height}")
    print(f"Number of segments: {len(clipped_segments)}")
    
    # Check if box is partially outside image
    is_partially_outside = (
        x_min < 0 or y_min < 0 or 
        x_min + width > image_width or 
        y_min + height > image_height
    )
    
    try:
        # Create a polygon from the clipped segments to get the actual visible area
        points = []
        for segment in clipped_segments:
            points.extend(segment)
        
        if len(points) < 3:
            print("Not enough points to form polygon")
            return 0.0
        
        # Convert points to numpy array
        points_array = np.array(points)
        
        # Find the point with the lowest y-coordinate (and leftmost if tied)
        start_idx = np.lexsort((points_array[:, 0], points_array[:, 1]))[0]
        start_point = points_array[start_idx]
        
        # Sort remaining points by angle with start point
        remaining_points = np.delete(points_array, start_idx, axis=0)
        angles = np.arctan2(remaining_points[:, 1] - start_point[1],
                          remaining_points[:, 0] - start_point[0])
        sorted_indices = np.argsort(angles)
        sorted_points = remaining_points[sorted_indices]
        
        # Combine start point with sorted points
        ordered_points = np.vstack((start_point, sorted_points))
        
        # Create polygon from ordered points
        visible_polygon = Polygon(ordered_points)
        
        if not visible_polygon.is_valid:
            print("Invalid polygon created from ordered points")
            return 0.0
        
        # Calculate visible area
        visible_area = visible_polygon.area
        
        # Calculate total area based on whether the box is truncated
        if is_partially_outside:
            # For truncated boxes, calculate the maximum possible visible area
            # by intersecting the box with image boundaries
            bbox_polygon = box(x_min, y_min, x_min + width, y_min + height)
            image_polygon = box(0, 0, image_width, image_height)
            max_visible_polygon = bbox_polygon.intersection(image_polygon)
            total_area = max_visible_polygon.area
        else:
            # For non-truncated boxes, use the actual box area
            total_area = width * height
        
        # Debug print
        print(f"\nDebug - Visibility Calculation:")
        print(f"Visible area: {visible_area}")
        print(f"Total area: {total_area}")
        
        # Calculate visibility percentage
        visibility = (visible_area / total_area) * 100
        
        # Clamp to 0-100 range
        visibility = max(0.0, min(100.0, visibility))
        
        print(f"Final visibility: {visibility}%")
        return visibility
        
    except Exception as e:
        print(f"Error in visibility calculation: {e}")
        return 0.0

def export_3d_bboxes(sensor_data, save_path, world, ego_vehicle, sensor_actor):
    """
    Exports 3D bounding box edges projected and clipped onto the camera image
    for DYNAMIC vehicles found via world.get_actors().
    """
    timestamp = int(sensor_data.timestamp * 1e3)
    image_w = float(sensor_data.width)
    image_h = float(sensor_data.height)
    fov = float(sensor_actor.attributes.get('fov', 90))

    K = build_projection_matrix(image_w, image_h, fov)
    w2c = np.array(sensor_actor.get_transform().get_inverse_matrix())
    sensor_transform = sensor_actor.get_transform()
    forward = sensor_transform.get_forward_vector()
    sensor_loc = sensor_transform.location
    ego_location = ego_vehicle.get_transform().location

    output_data = []

    # Process selected dynamic actors: vehicles and pedestrians
    vehicles = world.get_actors().filter('vehicle.*')
    walkers = world.get_actors().filter('walker.pedestrian.*')
    actors = list(vehicles) + list(walkers)
    for actor in actors:
        if actor.id == ego_vehicle.id:
            continue

        category = classify_actor_category(actor)
        if category is None or category not in EXPORT_BBOX3D_CATEGORIES:
            continue

        actor_transform = actor.get_transform()
        actor_loc = actor_transform.location

        # --- Basic Filters ---
        if actor_loc.distance(ego_location) > MAX_DISTANCE_METERS:
            continue
        ray = actor_loc - sensor_loc
        if forward.dot(ray) < 1:
            continue

        # --- Get Bounding Box ---
        try:
             bb = actor.bounding_box
             verts = list(bb.get_world_vertices(actor_transform))
             # Get bounding box dimensions
             extent = bb.extent
             # Convert to NuScenes format: [width, length, height]
             size = [extent.y * 2, extent.x * 2, extent.z * 2]  # CARLA x,y,z -> NuScenes width,length,height
        except AttributeError:
             print(f"Warning: Actor ID {actor.id} of type {actor.type_id} lacks 'bounding_box'. Skipping.")
             continue

        # --- Project Vertices ---
        projected_vertices = [get_image_point(v, K, w2c) for v in verts]

        # --- Clip Edges ---
        clipped_segments_for_actor = []
        all_projected_points_for_bbox = []
        for i, j in EDGES_INDICES:
            p1, behind1 = projected_vertices[i]
            p2, behind2 = projected_vertices[j]
            if behind1 and behind2:
                continue

            clip_result = liang_barsky_clip(
                p1[0], p1[1], p2[0], p2[1], 0.0, 0.0, image_w, image_h
            )
            if clip_result is not None:
                segment = [list(clip_result[0]), list(clip_result[1])]
                clipped_segments_for_actor.append(segment)
                all_projected_points_for_bbox.extend([segment[0], segment[1]])

        if not clipped_segments_for_actor:
            continue

        # --- Calculate Optional BBox ---
        bbox_from_clipped = None
        if all_projected_points_for_bbox:
            xs = [pt[0] for pt in all_projected_points_for_bbox]
            ys = [pt[1] for pt in all_projected_points_for_bbox]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w = max(0.0, x_max - x_min)
            h = max(0.0, y_max - y_min)
            bbox_from_clipped = [x_min, y_min, w, h]
            if w < MIN_BOX_PX or h < MIN_BOX_PX: # Tiny box filter
                 continue

        # --- Get Velocity ---
        velocity = actor.get_velocity()
        velocity_magnitude = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5

        # --- Save Actor Pose ---
        actor_pose = {
            "actor_id": actor.id,
            "timestamp": timestamp,
            "translation": {
                "x": actor_transform.location.x,
                "y": actor_transform.location.y,
                "z": actor_transform.location.z
            },
            "rotation": {
                "pitch": actor_transform.rotation.pitch,
                "yaw": actor_transform.rotation.yaw,
                "roll": actor_transform.rotation.roll
            }
        }

        # --- Store Data ---
        output_data.append({
            "actor_id": actor.id,
            "type": "vehicle" if category.startswith('vehicle.') else "pedestrian", 
            "category": category,
            "clipped_segments": clipped_segments_for_actor,
            "bbox_from_clipped": bbox_from_clipped,
            "velocity": {
                "x": velocity.x,
                "y": velocity.y,
                "z": velocity.z,
                "magnitude": velocity_magnitude
            },
            "pose": actor_pose,
            "size": size,  # Add bounding box dimensions
            "visibility": compute_visibility(clipped_segments_for_actor, bbox_from_clipped, image_w, image_h)
        })

    # --- Save to JSON ---
    output_file = os.path.join(save_path, f"{timestamp}_3dbbox.json")
    try:
        os.makedirs(save_path, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    except Exception as e:
        print(f"Error writing JSON file {output_file}: {e}")
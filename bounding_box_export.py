import os
import json
import numpy as np
import carla 

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

    # Process ONLY Dynamic Vehicles accessible via get_actors()
    vehicles = world.get_actors().filter('*vehicle*')
    for actor in vehicles:
        if actor.id == ego_vehicle.id:
            continue

        actor_transform = actor.get_transform()
        actor_loc = actor_transform.location

        # --- Basic Filters ---
        if actor_loc.distance(ego_location) > 50:
            continue
        ray = actor_loc - sensor_loc
        if forward.dot(ray) < 1:
            continue

        # --- Get Bounding Box ---
        try:
             bb = actor.bounding_box
             verts = list(bb.get_world_vertices(actor_transform))
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
            if w < 3 or h < 3: # Tiny box filter
                 continue

        # --- Store Data ---
        output_data.append({
            "actor_id": actor.id,
            "type": "vehicle", 
            "clipped_segments": clipped_segments_for_actor,
            "bbox_from_clipped": bbox_from_clipped
        })

    # --- Save to JSON ---
    output_file = os.path.join(save_path, f"{timestamp}_3dbbox.json")
    try:
        os.makedirs(save_path, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    except Exception as e:
        print(f"Error writing JSON file {output_file}: {e}")
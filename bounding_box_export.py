import os
import json
import numpy as np

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]

def export_3d_bboxes(sensor_data, save_path, world, ego_vehicle, sensor_actor):
    timestamp = int(sensor_data.timestamp * 1e3)
    image_w = sensor_data.width
    image_h = sensor_data.height
    fov = float(sensor_actor.attributes.get('fov', 90))
    
    K = build_projection_matrix(image_w, image_h, fov)
    w2c = np.array(sensor_actor.get_transform().get_inverse_matrix())
    sensor_transform = sensor_actor.get_transform()
    forward = sensor_transform.get_forward_vector()
    sensor_loc = sensor_transform.location

    bboxes = []
    for actor in world.get_actors().filter('*vehicle*'):
        if actor.id == ego_vehicle.id:
            continue
        actor_loc = actor.get_transform().location
        ray = actor_loc - sensor_loc
        # Only export if actor is clearly in front of the camera
        if forward.dot(ray) < 1:
            continue
        # Only annotate vehicles within 50m from ego vehicle
        if actor_loc.distance(ego_vehicle.get_transform().location) > 50:
            continue
        try:
            bb = actor.bounding_box
            verts = bb.get_world_vertices(actor.get_transform())
            projected = [get_image_point(v, K, w2c).tolist() for v in verts]
            # Clip projected points to image boundaries
            clipped = []
            for pt in projected:
                x = max(0, min(pt[0], image_w))
                y = max(0, min(pt[1], image_h))
                clipped.append([x, y])
            # Compute bounding rectangle from clipped vertices
            xs = [pt[0] for pt in clipped]
            ys = [pt[1] for pt in clipped]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            # Skip export if visible area is negligible (i.e. occlusion)
            if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                continue
            bboxes.append({
                "actor_id": actor.id,
                "type": "vehicle",
                "vertices": clipped,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min]
            })
        except Exception:
            continue
    
    output_file = os.path.join(save_path, f"{timestamp}_3dbbox.json")
    with open(output_file, 'w') as f:
        json.dump(bboxes, f, indent=2)
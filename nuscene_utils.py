import uuid
from typing import List, Any, Tuple
import numpy as np
from scipy.spatial.transform import Rotation
import carla

def generate_token() -> str:
    """Generate a unique token using UUID."""
    return uuid.uuid4().hex

def generate_composite_token(*args: Any) -> str:
    """Generate a composite token based on input arguments.

    Args:
        *args: Components to include in the composite token.

    Returns:
        A unique token string.
    """
    composite_string = "_".join(map(str, args))
    return uuid.uuid5(uuid.NAMESPACE_DNS, composite_string).hex

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> List[float]:
    """Convert Euler angles (roll, pitch, yaw) to a Quaternion (w, x, y, z).

    Args:
        roll: Rotation around the X-axis in degrees.
        pitch: Rotation around the Y-axis in degrees.
        yaw: Rotation around the Z-axis in degrees.

    Returns:
        A list representing the Quaternion [w, x, y, z].
    """
    # CARLA uses degrees, scipy expects radians
    quaternion = Rotation.from_euler('xyz', [np.radians(roll), np.radians(pitch), np.radians(yaw)]).as_quat()
    # NuScenes quaternion order is [w, x, y, z]
    # Scipy's as_quat() returns [x, y, z, w]
    return [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]


def convert_bounding_box_size(extent: carla.Vector3D) -> List[float]:
    """Convert CARLA bounding box extent to NuScenes size.

    Args:
        extent: A CARLA bounding box extent object with x, y, z (half-dimensions).

    Returns:
        A list representing the size [width, length, height] in NuScenes format.
    """
    # Double the extent values to get full dimensions
    width = extent.y * 2  # CARLA y -> NuScenes width
    length = extent.x * 2  # CARLA x -> NuScenes length
    height = extent.z * 2  # CARLA z -> NuScenes height
    return [width, length, height]

def transform_to_global_frame(transform: carla.Transform, global_origin: np.ndarray) -> dict:
    """Transform CARLA world coordinates to the global reference frame.

    Args:
        transform: A CARLA Transform object containing location and rotation.
        global_origin: The global origin [x, y, z] as a numpy array.

    Returns:
        A dictionary with `translation` and `rotation` in the global frame.
    """
    location = transform.location
    rotation = transform.rotation

    translation_np = np.array([location.x, location.y, location.z]) - global_origin

    # Scipy's from_euler expects radians. CARLA rotations are in degrees.
    # Order is roll, pitch, yaw.
    quat_scipy = Rotation.from_euler('xyz', [np.radians(rotation.roll), np.radians(rotation.pitch), np.radians(rotation.yaw)]).as_quat()
    # NuScenes quaternion order is [w, x, y, z]
    # Scipy's as_quat() returns [x, y, z, w]
    quaternion_nusc = [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]

    return {
        "translation": translation_np.tolist(),
        "rotation": quaternion_nusc
    }

def adjust_z_for_ego_pose(translation: List[float]) -> List[float]:
    """Adjust the Z-coordinate for ego_pose to match NuScenes' Z=0 assumption.

    Args:
        translation: A list representing the [x, y, z] translation.

    Returns:
        A modified translation with Z set to 0.
    """
    if len(translation) == 3:
        translation[2] = 0.0
    return translation

def is_point_in_box(point: np.ndarray, box_center: List[float], box_size: List[float], box_rotation: List[float]) -> bool:
    """Check if a point is inside an oriented 3D box.

    Args:
        point: [x, y, z] point in global frame
        box_center: [x, y, z] center of box in global frame
        box_size: [width, length, height] of box (NuScenes format)
        box_rotation: quaternion [w, x, y, z] of box rotation (NuScenes format)

    Returns:
        bool: True if point is inside box
    """
    point_local = point - np.array(box_center)

    # NuScenes quaternion [w, x, y, z] to Scipy [x, y, z, w]
    qw, qx, qy, qz = box_rotation
    r = Rotation.from_quat([qx, qy, qz, qw])
    
    # Transform point to box-local frame by rotating with the inverse of box_rotation
    # R.inv() gives the inverse rotation.
    point_local = r.inv().apply(point_local)

    # Box size in NuScenes is [width, length, height].
    # For local coordinate check, we need half dimensions along local x, y, z.
    # NuScenes: width (local y), length (local x), height (local z).
    half_width = box_size[0] / 2.0
    half_length = box_size[1] / 2.0
    half_height = box_size[2] / 2.0

    # Check if point is inside box bounds (local x, y, z)
    # Local x corresponds to length, local y to width, local z to height
    is_inside_x = abs(point_local[0]) <= half_length
    is_inside_y = abs(point_local[1]) <= half_width
    is_inside_z = abs(point_local[2]) <= half_height
    
    return is_inside_x and is_inside_y and is_inside_z

def transform_points_to_global(points: np.ndarray, sensor_transform: carla.Transform, ego_transform: carla.Transform) -> np.ndarray:
    """Transform points from sensor frame to global frame.
    
    Args:
        points: numpy array of points [N x 3] (x,y,z in sensor frame) or [N x 4] (with intensity)
        sensor_transform: CARLA transform of the sensor relative to ego
        ego_transform: CARLA transform of the ego vehicle in global CARLA frame
        
    Returns:
        Transformed points in global CARLA frame [N x 3] or [N x 4]
    """
    sensor_matrix = np.array(sensor_transform.get_matrix())
    ego_matrix = np.array(ego_transform.get_matrix())
    
    has_intensity = points.shape[1] == 4
    point_positions = points[:, :3]
    
    point_positions_h = np.hstack((point_positions, np.ones((len(points), 1))))
    
    # Transform points: sensor -> ego -> global
    points_ego_h = np.dot(point_positions_h, sensor_matrix.T)
    points_global_h = np.dot(points_ego_h, ego_matrix.T)
    
    points_global_cartesian = points_global_h[:, :3]

    if has_intensity:
        return np.column_stack((points_global_cartesian, points[:, 3]))
    return points_global_cartesian

def transform_radar_points_to_global(radar_points: np.ndarray, sensor_transform: carla.Transform, ego_transform: carla.Transform) -> np.ndarray:
    """Transform radar points from spherical to global CARLA frame.
    
    Args:
        radar_points: numpy array of radar points [N x 5] (depth, elevation, azimuth, velocity, intensity)
                      Angles are in degrees.
        sensor_transform: CARLA transform of the sensor relative to ego
        ego_transform: CARLA transform of the ego vehicle in global CARLA frame
            
    Returns:
        Transformed points in global CARLA frame [N x 3] (x,y,z)
    """
    if radar_points.size == 0:
        return np.array([])
    
    if radar_points.shape[1] != 5:
        # print(f"Warning: Unexpected radar points shape {radar_points.shape}, expected (N, 5)")
        return np.array([])
    
    cartesian_points_sensor_frame = []
    for point in radar_points:
        depth, elevation_deg, azimuth_deg, _, _ = point # velocity, intensity not used for position
        
        if not (-90 <= elevation_deg <= 90) or not (-180 <= azimuth_deg <= 180) or depth <= 0:
            # print(f"Warning: Invalid radar point data: d={depth}, el={elevation_deg}, az={azimuth_deg}")
            continue
            
        elevation_rad = np.radians(elevation_deg)
        azimuth_rad = np.radians(azimuth_deg)
        
        # CARLA radar sensor frame: X-forward, Y-right, Z-up
        # Spherical to Cartesian conversion:
        # x = depth * cos(elevation) * cos(azimuth)
        # y = depth * cos(elevation) * sin(azimuth)
        # z = depth * sin(elevation)
        x = depth * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = depth * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = depth * np.sin(elevation_rad)
        
        cartesian_points_sensor_frame.append([x, y, z])
    
    if not cartesian_points_sensor_frame:
        return np.array([])
    
    cartesian_points_sensor_frame_np = np.array(cartesian_points_sensor_frame)
    
    return transform_points_to_global(cartesian_points_sensor_frame_np, sensor_transform, ego_transform)

def count_points_in_box(points: np.ndarray, box_center: List[float], box_size: List[float], box_rotation: List[float]) -> int:
    """Count how many points fall inside a 3D box.
    
    Args:
        points: numpy array of points [N x 3] (x,y,z) or [N x 4] (with intensity) in global frame
        box_center: [x, y, z] center of box in global frame
        box_size: [width, length, height] of box (NuScenes format)
        box_rotation: quaternion [w, x, y, z] of box rotation (NuScenes format)
            
    Returns:
        int: Number of points inside box
    """
    point_positions = points[:, :3]
    count = 0
    for point in point_positions:
        if is_point_in_box(point, box_center, box_size, box_rotation):
            count += 1
    return count 
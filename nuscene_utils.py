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

def carla_rotation_to_nuscenes_quaternion(roll: float, pitch: float, yaw: float) -> List[float]:
    """Convert CARLA rotation to NuScenes quaternion using proper coordinate transformation.

    Args:
        roll: CARLA rotation around the X-axis in degrees.
        pitch: CARLA rotation around the Y-axis in degrees.
        yaw: CARLA rotation around the Z-axis in degrees.

    Returns:
        A list representing the NuScenes Quaternion [w, x, y, z].
    """
    # Create CARLA rotation matrix
    carla_rotation = Rotation.from_euler('xyz', [np.radians(roll), np.radians(pitch), np.radians(yaw)])
    carla_matrix = carla_rotation.as_matrix()
    
    # CARLA to NuScenes coordinate transformation matrix
    # CARLA: X-forward, Y-right, Z-up → NuScenes: X-forward, Y-left, Z-up
    transform_matrix = np.array([
        [1,  0,  0],
        [0,  -1,  0],
        [0,  0,  1]
    ])
    
    # Apply coordinate transformation: T * R_carla * T^(-1)
    # Since T is its own inverse: T * R_carla * T
    nuscenes_matrix = transform_matrix @ carla_matrix @ transform_matrix
    
    # Convert back to quaternion
    nuscenes_rotation = Rotation.from_matrix(nuscenes_matrix)
    quaternion = nuscenes_rotation.as_quat()
    
    # NuScenes quaternion order is [w, x, y, z]
    # Scipy's as_quat() returns [x, y, z, w]
    return [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]

def carla_camera_rotation_to_nuscenes_quaternion(roll: float, pitch: float, yaw: float) -> List[float]:
    """Convert CARLA camera rotation to NuScenes camera quaternion.
    
    For cameras, we need special handling because NuScenes camera coordinate system
    has Z pointing forward into the scene, while CARLA uses the standard coordinate system.

    Args:
        roll: CARLA rotation around the X-axis in degrees.
        pitch: CARLA rotation around the Y-axis in degrees.
        yaw: CARLA rotation around the Z-axis in degrees.

    Returns:
        A list representing the NuScenes Camera Quaternion [w, x, y, z].
    """
    # Convert CARLA Euler angles to rotation matrix
    carla_rotation = Rotation.from_euler('xyz', [np.radians(roll), np.radians(pitch), np.radians(yaw)])
    carla_matrix = carla_rotation.as_matrix()
    
    # CARLA to NuScenes camera coordinate transformation
    carla_to_camera = np.array([
        [0,  0,  1],  
        [0,  1,  0],  
        [-1,  0,  0]   
    ])
    
    # Apply the coordinate transformation to the CARLA rotation matrix
    nuscenes_camera_matrix = carla_to_camera @ carla_matrix @ carla_to_camera.T
    
    # Convert back to quaternion
    nuscenes_rotation = Rotation.from_matrix(nuscenes_camera_matrix)
    quaternion = nuscenes_rotation.as_quat()
    
    # NuScenes quaternion order is [w, x, y, z]
    # Scipy's as_quat() returns [x, y, z, w]
    return [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> List[float]:
    """Convert Euler angles (roll, pitch, yaw) to a Quaternion (w, x, y, z).
    
    This is the old function kept for backward compatibility.
    Use carla_rotation_to_nuscenes_quaternion for CARLA→NuScenes conversion.
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
    width = extent.y * 2   # CARLA y -> NuScenes width (Y)
    length = extent.x * 2  # CARLA x -> NuScenes length (X)
    height = extent.z * 2  # CARLA z -> NuScenes height (Z)
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

def transform_box_to_ego_frame(box_center_world: List[float], box_rotation_world: List[float], 
                              ego_translation: List[float], ego_rotation: List[float]) -> Tuple[List[float], List[float]]:
    """Transform bounding box from CARLA world coordinates to ego vehicle coordinates.
    
    Args:
        box_center_world: [x, y, z] center of box in CARLA world coordinates
        box_rotation_world: quaternion [w, x, y, z] of box rotation in world coordinates
        ego_translation: [x, y, z] ego vehicle position in CARLA world coordinates
        ego_rotation: quaternion [w, x, y, z] of ego vehicle rotation in world coordinates
        
    Returns:
        Tuple of (box_center_ego, box_rotation_ego) in ego vehicle coordinates
    """
    # Convert to numpy arrays
    box_center_world = np.array(box_center_world)
    ego_translation = np.array(ego_translation)
    
    # Get ego rotation matrix (world to ego)
    qw, qx, qy, qz = ego_rotation
    ego_rotation_matrix = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    
    # Transform box center: world -> ego
    # 1. Translate to ego-relative position
    box_center_relative = box_center_world - ego_translation
    # 2. Rotate to ego frame
    box_center_ego = ego_rotation_matrix.T @ box_center_relative

    # --- FIX: Flip Y axis to convert CARLA (Y-right) to nuScenes (Y-left) ---
    box_center_ego[1] *= -1
    
    # Transform box rotation: world -> ego
    # 1. Get world rotation matrix for box
    qw, qx, qy, qz = box_rotation_world
    box_rotation_matrix_world = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    # 2. Transform to ego frame: ego_rotation.T @ box_rotation_world
    box_rotation_matrix_ego = ego_rotation_matrix.T @ box_rotation_matrix_world
    # (No additional 180-degree rotation)
    # 3. Convert back to quaternion
    box_rotation_ego = Rotation.from_matrix(box_rotation_matrix_ego).as_quat()
    # Convert to nuScenes format [w, x, y, z]
    box_rotation_ego = [box_rotation_ego[3], box_rotation_ego[0], box_rotation_ego[1], box_rotation_ego[2]]
    
    return box_center_ego.tolist(), box_rotation_ego 
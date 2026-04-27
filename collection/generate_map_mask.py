#!/usr/bin/env python

import os
import sys
import time
import carla
import numpy as np
from PIL import Image
import yaml
import json
from pathlib import Path
import argparse

class MapMaskGenerator:
    """Generate map mask from CARLA semantic segmentation for NuScenes format."""
    
    def __init__(self, carla_host='localhost', carla_port=2000, output_dir='_out'):
        """Initialize the map mask generator.
        
        Args:
            carla_host: CARLA server hostname
            carla_port: CARLA server port
            output_dir: Directory to save the generated map mask
        """
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Camera parameters - start with lower resolution for testing
        self.camera_width = 1024  # Start with 1024x1024 for testing
        self.camera_height = 1024
        self.camera_fov = 90.0
        self.capture_altitude = 400.0  # meters above ground
        
        # NuScenes target resolution: 0.1 m/pixel
        self.target_resolution = 0.1  # meters per pixel
        
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        
    def connect_to_carla(self):
        """Connect to CARLA server and set up the world."""
        print(f"Connecting to CARLA server at {self.carla_host}:{self.carla_port}")
        self.client = carla.Client(self.carla_host, self.carla_port)
        self.client.set_timeout(10.0)
        
        # Get the world and current map
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        
        print(f"Connected to CARLA world: {self.map.name}")
        
        # Set synchronous mode for stable capture
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(settings)
        
    def get_map_bounds(self):
        """Calculate the bounds of the CARLA map."""
        print("Calculating map bounds...")
        
        # Get all waypoints to determine map boundaries
        waypoints = self.map.generate_waypoints(distance=10.0)  # Every 10 meters
        
        if not waypoints:
            print("Warning: No waypoints found, using default bounds")
            return [-500, -500, 500, 500]  # Default 1km x 1km area
        
        x_coords = [wp.transform.location.x for wp in waypoints]
        y_coords = [-wp.transform.location.y for wp in waypoints]  # Negate Y for NuScenes coordinate system
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Add some padding
        padding = 50.0  # 50 meter padding
        bounds = [min_x - padding, min_y - padding, max_x + padding, max_y + padding]
        
        print(f"Map bounds (NuScenes coords): X=[{bounds[0]:.1f}, {bounds[2]:.1f}], Y=[{bounds[1]:.1f}, {bounds[3]:.1f}]")
        print(f"Map size: {bounds[2] - bounds[0]:.1f}m x {bounds[3] - bounds[1]:.1f}m")
        print("Note: Y coordinates have been converted to NuScenes coordinate system (negated)")
        
        return bounds
        
    def calculate_capture_parameters(self, bounds):
        """Calculate camera parameters needed to cover the map area.
        
        Args:
            bounds: [min_x, min_y, max_x, max_y] of the map
            
        Returns:
            dict with capture parameters
        """
        map_width = bounds[2] - bounds[0]  # max_x - min_x
        map_height = bounds[3] - bounds[1]  # max_y - min_y
        
        # Calculate the area covered by camera at given altitude and FOV
        # Using FOV and altitude to calculate ground coverage
        fov_rad = np.radians(self.camera_fov)
        ground_width = 2 * self.capture_altitude * np.tan(fov_rad / 2)
        ground_height = ground_width  # Assuming square camera
        
        print(f"Camera covers {ground_width:.1f}m x {ground_height:.1f}m at {self.capture_altitude}m altitude")
        
        # Check if single capture can cover the entire map
        can_cover_in_single_shot = (ground_width >= map_width and ground_height >= map_height)
        
        if can_cover_in_single_shot:
            print("Map can be covered in a single capture!")
            # Position camera at map center (in NuScenes coordinates)
            center_x = (bounds[0] + bounds[2]) / 2
            center_y = (bounds[1] + bounds[3]) / 2
            # Convert back to CARLA coordinates for camera positioning (negate Y again)
            carla_center_x = center_x
            carla_center_y = -center_y  # Convert back to CARLA coordinate system for spawning
            capture_positions = [(carla_center_x, carla_center_y)]
            print(f"Camera position (CARLA coords): ({carla_center_x:.1f}, {carla_center_y:.1f})")
            print(f"This covers NuScenes area: ({center_x:.1f}, {center_y:.1f})")
        else:
            print("Map requires multiple captures - will implement stitching later")
            # For now, capture from center anyway as a test
            center_x = (bounds[0] + bounds[2]) / 2
            center_y = (bounds[1] + bounds[3]) / 2
            # Convert back to CARLA coordinates for camera positioning (negate Y again)  
            carla_center_x = center_x
            carla_center_y = -center_y  # Convert back to CARLA coordinate system for spawning
            capture_positions = [(carla_center_x, carla_center_y)]
            print(f"Camera position (CARLA coords): ({carla_center_x:.1f}, {carla_center_y:.1f})")
            print(f"This covers NuScenes area: ({center_x:.1f}, {center_y:.1f})")
            
        return {
            'positions': capture_positions,
            'ground_coverage': (ground_width, ground_height),
            'map_size': (map_width, map_height),
            'single_shot': can_cover_in_single_shot
        }
    
    def spawn_vehicle_and_camera(self, x, y):
        """Spawn vehicle and attach semantic camera at specified position.
        
        Args:
            x, y: World coordinates for vehicle position
        """
        print(f"Spawning vehicle at ({x:.1f}, {y:.1f})")
        
        # Get vehicle blueprint
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]  # Get any vehicle
        
        # Find a spawn point near the desired location
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available on this map")
            
        # Find closest spawn point to desired location
        target_location = carla.Location(x=x, y=y, z=0)
        closest_spawn = min(spawn_points, 
                           key=lambda sp: sp.location.distance(target_location))
        
        # Spawn vehicle
        self.vehicle = self.world.spawn_actor(vehicle_bp, closest_spawn)
        print(f"Vehicle spawned at {self.vehicle.get_location()}")
        
        # Create semantic camera
        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', str(self.camera_width))
        camera_bp.set_attribute('image_size_y', str(self.camera_height))
        camera_bp.set_attribute('fov', str(self.camera_fov))
        
        # Position camera high above vehicle, looking straight down
        camera_transform = carla.Transform(
            carla.Location(x=0, y=0, z=self.capture_altitude),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
        )
        
        # Spawn camera attached to vehicle
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        print(f"Semantic camera spawned at altitude {self.capture_altitude}m")
        
    def capture_semantic_image(self):
        """Capture semantic segmentation image.
        
        Returns:
            CARLA semantic image object
        """
        print("Capturing semantic image...")
        
        captured_image = None
        
        def image_callback(image):
            nonlocal captured_image
            captured_image = image
            
        # Set up callback
        self.camera.listen(image_callback)
        
        # Wait for capture
        timeout = 10.0  # 10 second timeout
        start_time = time.time()
        
        while captured_image is None and (time.time() - start_time) < timeout:
            self.world.tick()
            time.sleep(0.1)
            
        if captured_image is None:
            raise RuntimeError("Failed to capture semantic image within timeout")
            
        # Stop listening
        self.camera.stop()
        
        print("Semantic image captured successfully")
        return captured_image
        
    def process_semantic_to_nuscenes_mask(self, semantic_image, save_path, json_path, bounds, capture_pos):
        """Process semantic image to create NuScenes-format map mask.
        
        Args:
            semantic_image: CARLA semantic image
            save_path: Path to save the processed mask
            json_path: Path to save the json metadata for the mask
            bounds: Map bounds [min_x, min_y, max_x, max_y]
            capture_pos: (x, y) position where image was captured
        """
        print("Processing semantic image to NuScenes map mask...")
        
        # Convert semantic image to CityScapes palette to get colored version
        # Create temporary file for palette image
        temp_palette_path = save_path.parent / "temp_palette.png"
        semantic_image.save_to_disk(str(temp_palette_path), carla.ColorConverter.CityScapesPalette)
        
        # Load the palette image as RGB
        palette_image = Image.open(temp_palette_path).convert('RGB')
        palette_array = np.array(palette_image)
        
        print(f"Loaded palette image with shape: {palette_array.shape}")
        
        # Define the target colors in CityScapes palette
        # CARLA Road (label 7) maps to RGB(128, 64, 128) in CityScapes palette
        # CARLA RoadLine (label 6) maps to RGB(157, 234, 50) in CityScapes palette
        ROAD_COLOR_RGB = (128, 64, 128)
        ROADLINE_COLOR_RGB = (157, 234, 50)
        
        # Create mask for road pixels
        road_mask = np.all(palette_array == ROAD_COLOR_RGB, axis=2)
        roadline_mask = np.all(palette_array == ROADLINE_COLOR_RGB, axis=2)
        
        # Combine both masks - both roads and road lines should be drivable
        drivable_mask = road_mask | roadline_mask
        
        road_pixel_count = np.sum(road_mask)
        roadline_pixel_count = np.sum(roadline_mask)
        total_drivable_count = np.sum(drivable_mask)
        total_pixels = palette_array.shape[0] * palette_array.shape[1]
        
        print(f"Found road pixels: {road_pixel_count} ({road_pixel_count/total_pixels*100:.1f}%)")
        print(f"Found road line pixels: {roadline_pixel_count} ({roadline_pixel_count/total_pixels*100:.1f}%)")
        print(f"Total drivable pixels: {total_drivable_count} ({total_drivable_count/total_pixels*100:.1f}%)")
        
        # Create NuScenes-format map mask
        # Drivable areas (roads + road lines): RGB(250, 250, 250) - light gray  
        # Non-drivable areas: RGB(200, 200, 200) - darker gray
        nuscenes_mask = np.full((palette_array.shape[0], palette_array.shape[1], 3), 
                               (200, 200, 200), dtype=np.uint8)  # Default: non-drivable color
        nuscenes_mask[drivable_mask] = (250, 250, 250)  # Drivable color (roads + road lines)
        
        print("Created NuScenes-format mask with road and road line colors")
        
        # === RESOLUTION SCALING TO 0.1 m/pixel ===
        
        # Calculate real-world area covered by camera
        fov_rad = np.radians(self.camera_fov)
        ground_width = 2 * self.capture_altitude * np.tan(fov_rad / 2)
        ground_height = ground_width  # Square FOV
        
        print(f"Ground coverage: {ground_width:.1f}m x {ground_height:.1f}m")
        print(f"Original resolution: {ground_width/self.camera_width:.3f} m/pixel")
        
        # Calculate target image size for target resolution
        target_width_pixels = int(ground_width / self.target_resolution)
        target_height_pixels = int(ground_height / self.target_resolution)
        
        print(f"Target size for {self.target_resolution} m/pixel: {target_width_pixels}x{target_height_pixels}")
        
        # Scale the mask to target resolution
        if (target_width_pixels != self.camera_width or target_height_pixels != self.camera_height):
            print(f"Resampling from {self.camera_width}x{self.camera_height} to {target_width_pixels}x{target_height_pixels}")
            
            # Use PIL for high-quality resampling
            mask_pil = Image.fromarray(nuscenes_mask, mode='RGB')
            
            # Use LANCZOS for high-quality downsampling, BICUBIC for upsampling
            if target_width_pixels < self.camera_width:
                resample_method = Image.Resampling.LANCZOS
            else:
                resample_method = Image.Resampling.BICUBIC
                
            scaled_mask_pil = mask_pil.resize(
                (target_width_pixels, target_height_pixels), 
                resample=resample_method
            )
            
            # Convert back to numpy array
            scaled_nuscenes_mask = np.array(scaled_mask_pil)
            
            print(f"Scaling completed. New dimensions: {scaled_nuscenes_mask.shape}")
            
        else:
            print("No scaling needed - already at target resolution")
            scaled_nuscenes_mask = nuscenes_mask
            
        # Flip the image vertically to match NuScenes coordinate system (Y negated)
        scaled_nuscenes_mask_flipped = np.flipud(scaled_nuscenes_mask)

        # Rotate 90° CCW so pixel axes align with nuScenes (X,Y) convention
        scaled_nuscenes_mask_flipped = np.rot90(scaled_nuscenes_mask_flipped, k=1)
        
        # Save the final NuScenes map mask
        scaled_mask_image = Image.fromarray(scaled_nuscenes_mask_flipped, mode='RGB')
        scaled_mask_image.save(save_path)
        print(f"NuScenes map mask saved to {save_path} (Y-flipped for coordinate system alignment)")

        # === WRITE COMPANION METADATA JSON FOR NUSCENES MAP API ===
        try:
            map_name = save_path.stem.replace('_basemap', '') # e.g. Town10HD_Opt
            
            # Use the original CARLA map name to preserve map identity
            # Create NuScenesMap-compatible metadata
            meta = {
                "map_name": map_name,  # Use original CARLA map name
                "original_carla_map": map_name,  # Keep track of original name
                "dataroot": "./",  # Relative to nuscenes dataset root
                "origin": [float(bounds[0]), float(bounds[3])],  # World coordinate of top-left corner (min_x, max_y)  
                "scale": float(self.target_resolution),
                "rotation": 0.0,
                "basemap": {
                    "filename": save_path.name,
                    "resolution": float(self.target_resolution),
                    "size": [scaled_nuscenes_mask_flipped.shape[1], scaled_nuscenes_mask_flipped.shape[0]]
                }
            }

            with open(json_path, 'w') as jf:
                json.dump(meta, jf, indent=2)
            print(f"NuScenes map metadata written to {json_path}")
        except Exception as e:
            print(f"Warning: Failed to write metadata json: {e}")
        
        # Calculate final resolution achieved
        final_resolution = ground_width / scaled_nuscenes_mask.shape[1]
        print(f"Final resolution achieved: {final_resolution:.4f} m/pixel")
        
        # Clean up temporary file
        temp_palette_path.unlink()
        
        return scaled_nuscenes_mask
        
    def cleanup(self):
        """Clean up spawned actors."""
        print("Cleaning up...")
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
            
        # Restore async mode
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
    def generate_map_mask(self):
        """Main function to generate the map mask."""
        try:
            # Connect to CARLA
            self.connect_to_carla()
            
            # Calculate map bounds and capture parameters
            bounds = self.get_map_bounds()
            capture_params = self.calculate_capture_parameters(bounds)
            
            # Generate filename based on map name
            map_name = self.map.name.split('/')[-1]  # Extract map name
            timestamp = int(time.time())
            
            # Process each capture position
            for i, (x, y) in enumerate(capture_params['positions']):
                print(f"\n=== Capture {i+1}/{len(capture_params['positions'])} ===")
                
                # Clean up previous actors if any
                if self.camera:
                    self.camera.destroy()
                if self.vehicle:
                    self.vehicle.destroy()
                    
                # Spawn vehicle and camera at position
                self.spawn_vehicle_and_camera(x, y)
                
                # Wait for stabilization
                for _ in range(10):
                    self.world.tick()
                    
                # Capture semantic image
                semantic_image = self.capture_semantic_image()
                
                # Process to NuScenes map mask
                map_name = self.map.name.split('/')[-1]
                basemap_name = f"{map_name}_basemap.png"
                mask_path = self.output_dir / basemap_name
                json_path = self.output_dir / f"{map_name}.json"
                
                self.process_semantic_to_nuscenes_mask(semantic_image, mask_path, json_path, bounds, (x, y))
                
            print(f"\n=== Map mask generation completed! ===")
            print(f"Generated NuScenes map mask in: {self.output_dir}")
            print(f"Map files created: {list(self.output_dir.glob('*.png'))}")
            
            if not capture_params['single_shot']:
                print("Note: Map requires multiple captures. Stitching implementation needed for complete coverage.")
                
        except Exception as e:
            print(f"Error during map mask generation: {e}")
            raise
        finally:
            self.cleanup()

def main():
    parser = argparse.ArgumentParser(description='Generate map mask for NuScenes from CARLA semantic segmentation')
    parser.add_argument('--host', default='localhost', help='CARLA server hostname')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--output', default='_out', help='Output directory for map masks')
    parser.add_argument('--resolution', choices=['low', 'medium', 'high'], default='low',
                       help='Camera resolution: low=1024x1024, medium=2048x2048, high=4096x4096')
    parser.add_argument('--altitude', type=float, default=400.0, help='Camera altitude in meters')
    parser.add_argument('--target-resolution', type=float, default=0.1, 
                       help='Target resolution in meters per pixel (default: 0.1 for NuScenes standard)')
    
    args = parser.parse_args()
    
    # Set resolution based on argument
    resolution_map = {
        'low': 1024,
        'medium': 2048, 
        'high': 4096
    }
    
    generator = MapMaskGenerator(
        carla_host=args.host,
        carla_port=args.port,
        output_dir=args.output
    )
    
    # Update camera resolution and target resolution
    generator.camera_width = resolution_map[args.resolution]
    generator.camera_height = resolution_map[args.resolution]
    generator.capture_altitude = args.altitude
    generator.target_resolution = args.target_resolution
    
    print(f"Starting map mask generation:")
    print(f"  Camera resolution: {args.resolution} ({generator.camera_width}x{generator.camera_height})")
    print(f"  Altitude: {args.altitude}m")
    print(f"  Target resolution: {args.target_resolution} m/pixel")
    
    generator.generate_map_mask()

if __name__ == '__main__':
    main()
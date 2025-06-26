#!/usr/bin/env python

import os
import json
import numpy as np
from pathlib import Path
import shutil
from PIL import Image
import struct

class NuScenesFixes:
    """Comprehensive fix module for nuScenes data format issues."""
    
    def __init__(self, output_base: Path):
        self.output_base = output_base
        self.version_dir = output_base / 'v1.0'
        
    def fix_all_issues(self):
        """Apply all fixes to ensure nuScenes compatibility."""
        print("\n=== Applying nuScenes Format Fixes ===")
        
        # 1. Fix file formats
        self.fix_file_formats()
        
        # 2. Fix LIDAR channel name
        self.fix_lidar_channel_name()
        
        # 3. Fix sample data mapping
        self.fix_sample_data_mapping()
        
        # 4. Fix camera intrinsics
        self.fix_camera_intrinsics()
        
        # 5. Fix LIDAR data quality
        self.fix_lidar_data_quality()
        
        # 6. Fix map file
        self.fix_map_file()
        
        # 7. Clean up empty directories
        self.cleanup_empty_directories()
        
        print("=== All fixes applied successfully! ===\n")
    
    def fix_file_formats(self):
        """Convert files to proper nuScenes formats."""
        print("1. Converting file formats...")
        
        # Create sensor directories (only for sensors that exist in nuScenes)
        sensor_dirs = {
            'Camera_Front': 'samples/CAM_FRONT',
            'Camera_Back': 'samples/CAM_BACK',
            'Camera_FrontRight': 'samples/CAM_FRONTRIGHT',
            'Camera_FrontLeft': 'samples/CAM_FRONTLEFT',
            'Camera_BackRight': 'samples/CAM_BACKRIGHT',
            'Camera_BackLeft': 'samples/CAM_BACKLEFT',
            'Lidar': 'samples/LIDAR_TOP',
            'Radar_Front': 'samples/RADAR_FRONT',
            'Radar_FrontRight': 'samples/RADAR_FRONTRIGHT',
            'Radar_FrontLeft': 'samples/RADAR_FRONTLEFT',
            'Radar_BackRight': 'samples/RADAR_BACKRIGHT',
            'Radar_BackLeft': 'samples/RADAR_BACKLEFT',
        }
        
        # Create directories
        for sensor_dir in sensor_dirs.values():
            (self.output_base / sensor_dir).mkdir(parents=True, exist_ok=True)
            sweep_dir = sensor_dir.replace('samples/', 'sweeps/')
            (self.output_base / sweep_dir).mkdir(parents=True, exist_ok=True)
        
        # Convert files from _out directory
        input_base = Path('_out')
        if not input_base.exists():
            print("Warning: _out directory not found, skipping file conversion")
            return
        
        files_converted = 0
        
        for scene_folder in input_base.iterdir():
            if not scene_folder.is_dir() or not scene_folder.name.startswith('scene_'):
                continue
                
            print(f"  Processing scene: {scene_folder.name}")
            
            for sensor_name, target_dir in sensor_dirs.items():
                sensor_path = scene_folder / sensor_name
                if not sensor_path.exists():
                    # This is normal for sensors that don't exist in the data
                    continue
                
                target_path = self.output_base / target_dir
                
                # Convert files based on sensor type
                if sensor_name.startswith('Camera'):
                    files_converted += self._convert_camera_files(sensor_path, target_path)
                elif sensor_name == 'Lidar':
                    files_converted += self._convert_lidar_files(sensor_path, target_path)
                elif sensor_name.startswith('Radar'):
                    files_converted += self._convert_radar_files(sensor_path, target_path)
        
        print(f"  Converted {files_converted} files")
    
    def _convert_camera_files(self, source_path: Path, target_path: Path) -> int:
        """Convert camera PNG files to JPG."""
        converted = 0
        
        for png_file in source_path.glob('*.png'):
            try:
                timestamp = png_file.stem.split('_')[0]
                target_file = target_path / f"{timestamp}.jpg"
                
                # Convert PNG to JPG
                img = Image.open(png_file)
                img = img.convert('RGB')
                img.save(target_file, 'JPEG', quality=95)
                converted += 1
                
            except Exception as e:
                print(f"    Error converting {png_file.name}: {e}")
        
        return converted
    
    def _convert_lidar_files(self, source_path: Path, target_path: Path) -> int:
        """Convert LIDAR NPY files to BIN format."""
        converted = 0
        
        for npy_file in source_path.glob('*.npy'):
            try:
                timestamp = npy_file.stem.split('_')[0]
                target_file = target_path / f"{timestamp}.bin"
                
                # Load and convert NPY to BIN
                points = np.load(npy_file)
                
                # Ensure 5-column format (x, y, z, intensity, ring_index)
                if points.shape[1] < 5:
                    padded_points = np.zeros((points.shape[0], 5))
                    padded_points[:, :points.shape[1]] = points
                    points = padded_points
                
                # Save as binary
                points.astype(np.float32).tofile(target_file)
                converted += 1
                
            except Exception as e:
                print(f"    Error converting {npy_file.name}: {e}")
        
        return converted
    
    def _convert_radar_files(self, source_path: Path, target_path: Path) -> int:
        """Convert RADAR NPY files to binary PCD format."""
        converted = 0
        
        for npy_file in source_path.glob('*.npy'):
            try:
                timestamp = npy_file.stem.split('_')[0]
                target_file = target_path / f"{timestamp}.pcd"
                
                # Load NPY data
                points = np.load(npy_file)
                
                # Ensure 18-column format for RADAR
                if points.shape[1] < 18:
                    padded_points = np.zeros((points.shape[0], 18))
                    padded_points[:, :points.shape[1]] = points
                    points = padded_points
                
                # Write binary PCD file
                with open(target_file, 'wb') as f:
                    # Write PCD header
                    f.write(b"# .PCD v0.7 - Point Cloud Data file format\n")
                    f.write(b"VERSION 0.7\n")
                    f.write(b"FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms\n")
                    f.write(b"SIZES 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1\n")
                    f.write(b"TYPES F F F I I F F F F F I I I I I I I I\n")
                    f.write(b"COUNTS 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n")
                    f.write(f"WIDTH {points.shape[0]}\n".encode())
                    f.write(b"HEIGHT 1\n")
                    f.write(b"VIEWPOINT 0 0 0 1 0 0 0\n")
                    f.write(f"POINTS {points.shape[0]}\n".encode())
                    f.write(b"DATA binary\n")
                    
                    # Write binary data
                    points.astype(np.float32).tofile(f)
                
                converted += 1
                
            except Exception as e:
                print(f"    Error converting {npy_file.name}: {e}")
        
        return converted
    
    def fix_lidar_channel_name(self):
        """Fix LIDAR channel name in sensor.json."""
        print("2. Fixing LIDAR channel name...")
        
        sensor_file = self.version_dir / 'sensor.json'
        if not sensor_file.exists():
            print("  Warning: sensor.json not found")
            return
        
        with open(sensor_file, 'r') as f:
            sensors = json.load(f)
        
        # Check if LIDAR channel needs updating
        lidar_found = False
        for sensor in sensors:
            if sensor['channel'] == 'LIDAR':
                sensor['channel'] = 'LIDAR_TOP'
                print("  Updated LIDAR channel to LIDAR_TOP")
                lidar_found = True
                break
            elif sensor['channel'] == 'LIDAR_TOP':
                print("  LIDAR channel already correctly set to LIDAR_TOP")
                lidar_found = True
                break
        
        if not lidar_found:
            print("  Warning: No LIDAR sensor found in sensor.json")
            return
        
        with open(sensor_file, 'w') as f:
            json.dump(sensors, f, indent=2)
    
    def fix_sample_data_mapping(self):
        """Fix sample data mapping in sample.json."""
        print("3. Fixing sample data mapping...")
        
        sample_file = self.version_dir / 'sample.json'
        sample_data_file = self.version_dir / 'sample_data.json'
        
        if not sample_file.exists() or not sample_data_file.exists():
            print("  Warning: sample.json or sample_data.json not found")
            return
        
        with open(sample_file, 'r') as f:
            samples = json.load(f)
        with open(sample_data_file, 'r') as f:
            sample_data = json.load(f)
        
        # Create lookup for sample_data by sample_token and channel
        sample_data_lookup = {}
        for sd in sample_data:
            sample_token = sd['sample_token']
            if sample_token not in sample_data_lookup:
                sample_data_lookup[sample_token] = {}
            
            # Extract channel from filename
            filename = sd['filename']
            if '/' in filename:
                channel = filename.split('/')[1]  # samples/CAM_FRONT/123.jpg -> CAM_FRONT
                sample_data_lookup[sample_token][channel] = sd['token']
        
        # Update samples with data field
        updated_count = 0
        for sample in samples:
            sample_token = sample['token']
            if sample_token in sample_data_lookup:
                sample['data'] = sample_data_lookup[sample_token]
                updated_count += 1
        
        with open(sample_file, 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f"  Updated {updated_count} samples with data field mappings")
    
    def fix_camera_intrinsics(self):
        """Fix camera intrinsic matrices in calibrated_sensor.json."""
        print("4. Fixing camera intrinsic matrices...")
        
        calibrated_file = self.version_dir / 'calibrated_sensor.json'
        sensor_file = self.version_dir / 'sensor.json'
        
        if not calibrated_file.exists() or not sensor_file.exists():
            print("  Warning: calibrated_sensor.json or sensor.json not found")
            return
        
        with open(calibrated_file, 'r') as f:
            calibrated_sensors = json.load(f)
        with open(sensor_file, 'r') as f:
            sensors = json.load(f)
        
        # Create mapping from sensor_token to modality
        sensor_token_to_modality = {s['token']: s['modality'] for s in sensors}
        
        # Standard camera intrinsic matrix (placeholder values)
        # These should be replaced with actual camera calibration data
        standard_intrinsics = [
            [1000.0, 0.0, 960.0],
            [0.0, 1000.0, 540.0],
            [0.0, 0.0, 1.0]
        ]
        
        fixed_count = 0
        for calibrated_sensor in calibrated_sensors:
            sensor_token = calibrated_sensor.get('sensor_token')
            if sensor_token and sensor_token_to_modality.get(sensor_token) == 'camera':
                if not calibrated_sensor.get('camera_intrinsic') or calibrated_sensor['camera_intrinsic'] == []:
                    calibrated_sensor['camera_intrinsic'] = standard_intrinsics
                    fixed_count += 1
        
        with open(calibrated_file, 'w') as f:
            json.dump(calibrated_sensors, f, indent=2)
        
        print(f"  Fixed camera intrinsic matrices for {fixed_count} camera sensors")
    
    def fix_lidar_data_quality(self):
        """Fix LIDAR data quality issues."""
        print("5. Fixing LIDAR data quality...")
        
        # Check for LIDAR files with problematic Z values
        lidar_dir = self.output_base / 'samples/LIDAR_TOP'
        if not lidar_dir.exists():
            print("  Warning: LIDAR_TOP directory not found")
            return
        
        removed_count = 0
        for bin_file in lidar_dir.glob('*.bin'):
            try:
                # Load binary data
                data = np.fromfile(bin_file, dtype=np.float32)
                if data.size % 5 == 0:
                    points = data.reshape(-1, 5)
                    
                    # Check for points with very small Z values (likely ground plane issues)
                    small_z_mask = np.abs(points[:, 2]) < 0.01
                    if np.any(small_z_mask):
                        # Remove points with very small Z values
                        filtered_points = points[~small_z_mask]
                        if len(filtered_points) > 0:
                            filtered_points.astype(np.float32).tofile(bin_file)
                            removed_count += np.sum(small_z_mask)
                        else:
                            # If all points are problematic, remove the file
                            bin_file.unlink()
                            print(f"    Removed problematic file: {bin_file.name}")
                
            except Exception as e:
                print(f"    Error processing {bin_file.name}: {e}")
        
        print(f"  Removed {removed_count} problematic LIDAR points")
    
    def fix_map_file(self):
        """Fix map file reference."""
        print("6. Fixing map file...")
        
        map_file = self.version_dir / 'map.json'
        if not map_file.exists():
            print("  Warning: map.json not found")
            return
        
        with open(map_file, 'r') as f:
            maps = json.load(f)
        
        # Update map filename to point to the correct file
        if maps:
            maps[0]['filename'] = 'maps/none.png'
        
        with open(map_file, 'w') as f:
            json.dump(maps, f, indent=2)
        
        print("  Updated map.json with correct filename")
    
    def cleanup_empty_directories(self):
        """Remove empty sweeps directories and other empty folders."""
        print("7. Cleaning up empty directories...")
        
        # Remove empty sweeps directories (normal for static datasets)
        sweeps_dir = self.output_base / 'sweeps'
        if sweeps_dir.exists():
            for sensor_dir in sweeps_dir.iterdir():
                if sensor_dir.is_dir():
                    try:
                        # Remove empty sensor directories in sweeps
                        if not any(sensor_dir.iterdir()):
                            shutil.rmtree(sensor_dir)
                            print(f"  Removed empty sweeps directory: {sensor_dir.name}")
                    except Exception as e:
                        print(f"  Error removing {sensor_dir}: {e}")
        
        # Remove empty sample directories
        samples_dir = self.output_base / 'samples'
        if samples_dir.exists():
            for sensor_dir in samples_dir.iterdir():
                if sensor_dir.is_dir():
                    try:
                        # Remove empty sensor directories in samples
                        if not any(sensor_dir.iterdir()):
                            shutil.rmtree(sensor_dir)
                            print(f"  Removed empty samples directory: {sensor_dir.name}")
                    except Exception as e:
                        print(f"  Error removing {sensor_dir}: {e}")
        
        print("  Cleanup completed") 
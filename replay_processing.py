import os
import sys
import numpy as np
import pygame
import time
from pathlib import Path
import math
import yaml
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

def process_camera(file_path, sensor_name, annotation_type, cell_size):
    try:
        # Charger l'image
        img = pygame.image.load(str(file_path))
        
        # Choose annotation file based on annotation_type setting
        if annotation_type == "3d":
            bbox_file = str(file_path).replace('.png', '_3dbbox.json')
        else:
            bbox_file = str(file_path).replace('.png', '_bbox.json')
            
        if os.path.exists(bbox_file):
            try:
                with open(bbox_file, 'r') as f:
                    bbox_data = json.load(f)
                
                # Determine bounding boxes list from loaded data
                if isinstance(bbox_data, dict):
                    boxes = bbox_data.get('bounding_boxes', [])
                else:
                    boxes = bbox_data
                    
                # Create a surface from the image
                surface = pygame.Surface(img.get_size(), pygame.SRCALPHA)
                surface.blit(img, (0, 0))
                
                if annotation_type == "3d":
                    edges = [[0,1], [1,3], [3,2], [2,0],
                                [0,4], [4,5], [5,1], [5,7],
                                [7,6], [6,4], [6,2], [7,3]]
                    img_width, img_height = img.get_size()
                    for bbox in boxes:
                        if 'vertices' in bbox:
                            verts = bbox['vertices']
                            xs = [v[0] for v in verts]
                            ys = [v[1] for v in verts]
                            # Compute the center of the projected box
                            avg_x = sum(xs) / len(xs)
                            avg_y = sum(ys) / len(ys)
                            # Only draw if the box center is within the image
                            if not (0 <= avg_x < img_width and 0 <= avg_y < img_height):
                                continue
                            for edge in edges:
                                start = verts[edge[0]]
                                end = verts[edge[1]]
                                pygame.draw.line(surface, (255, 0, 0),
                                                    (int(start[0]), int(start[1])),
                                                    (int(end[0]), int(end[1])), 2)
                            if 'distance' in bbox:
                                font = pygame.font.Font(None, 24)
                                dist_text = font.render(f"{bbox['distance']:.1f}m", True, (255,255,255))
                                surface.blit(dist_text, (int(verts[0][0]), int(verts[0][1])-20))
                    return surface
                else:
                    # Existing 2D drawing: draw axis-aligned rectangle
                    for bbox in boxes:
                        if 'bbox' in bbox:
                            x, y, w, h = bbox['bbox']
                        elif 'vertices' in bbox:
                            verts = bbox['vertices']
                            xs = [v[0] for v in verts]
                            ys = [v[1] for v in verts]
                            x_min = min(xs)
                            y_min = min(ys)
                            x_max = max(xs)
                            y_max = max(ys)
                            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
                        else:
                            continue
                        pygame.draw.rect(surface, (255, 0, 0), (x, y, w, h), 2)
                        if 'distance' in bbox:
                            font = pygame.font.Font(None, 24)
                            dist_text = font.render(f"{bbox['distance']:.1f}m", True, (255,255,255))
                            surface.blit(dist_text, (x, y-20))
                    return surface
            except Exception as e:
                print(f"Error loading bbox data: {e}")
                return img
        return img
    except Exception as e:
        print(f"Error loading camera image: {e}")
        return pygame.Surface(cell_size)

def process_radar(file_path, cell_size):
    try:
        data = np.load(file_path)
        # Create range-doppler map
        v_bins = 128
        r_bins = 128
        max_range = 250
        max_velocity = 30
        intensity_min = -70
        intensity_max = 0
        intensity_matrix = np.zeros((r_bins, v_bins))
        for point in data:
            depth, _, _, velocity, intensity = point
            r_idx = int((depth / max_range) * (r_bins - 1))
            v_idx = int(((velocity + max_velocity) / (2 * max_velocity)) * (v_bins - 1))
            if 0 <= r_idx < r_bins and 0 <= v_idx < v_bins:
                intensity_matrix[r_idx, v_idx] += intensity
        range_doppler = 20 * np.log10(intensity_matrix + 1e-10)
        
        # Use a larger figure (8x6) to improve resolution and text clarity
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8,6))  # Increased figure size
        fig.patch.set_facecolor('black')
        im = ax.imshow(range_doppler, aspect='auto', origin='lower', cmap='jet',
                        extent=[-max_velocity, max_velocity, 0, max_range],
                        vmin=intensity_min, vmax=intensity_max)
        cbar = fig.colorbar(im)
        cbar.ax.tick_params(colors='white', labelsize=8)
        cbar.set_label('Intensity (dB)', color='white', fontsize=8)
        ax.set_title('Range-Doppler Map', color='white', pad=10, fontsize=10)
        ax.set_xlabel('Velocity (m/s)', color='white', labelpad=8, fontsize=8)
        ax.set_ylabel('Range (m)', color='white', labelpad=8, fontsize=8)
        ax.tick_params(axis='both', colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(0.5)
        fig.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)
        
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        arr = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        # Avoid excessive scaling here to prevent pixelation
        return pygame.surfarray.make_surface(arr.swapaxes(0, 1))
    except Exception as e:
        print(f"Error processing radar file {file_path.name}: {e}")
        return pygame.Surface(cell_size)

def process_lidar(file_path, cell_size):
    try:
        points = np.load(file_path)
        # Create a blank image for display
        width, height = cell_size
        lidar_range = 100.0
        lidar_data = np.array(points[:, :2])
        scale = min(width, height) / lidar_range
        lidar_data = lidar_data * scale
        # Center the data
        lidar_data += (width/2, height/2)
        # Invert the y-coordinates for correct orientation
        lidar_data[:, 1] = height - lidar_data[:, 1] - 1
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.clip(lidar_data, 0, [width-1, height-1])
        lidar_img = np.zeros((height, width, 3), dtype=np.uint8)
        lidar_img[lidar_data[:, 1], lidar_data[:, 0]] = (255, 255, 255)
        surface = pygame.surfarray.make_surface(lidar_img)
        # Rotate the surface 90° to the left
        return pygame.transform.rotate(surface, 90)
    except Exception as e:
        print(f"Error processing lidar file {file_path.name}: {e}")
        return pygame.Surface(cell_size)

def process_semantic_lidar(file_path, cell_size, semantic_colors):
    try:
        # Load the semantic LIDAR data
        points = np.load(file_path)
        
        # Create a blank image for display
        width, height = cell_size
        lidar_range = 100.0
        
        # Extract XY coordinates and semantic tags
        lidar_data = np.array(points[['x', 'y']].tolist())
        semantic_tags = points['semantic_tag']
        
        # Scale and center the data
        scale = min(width, height) / lidar_range
        lidar_data = lidar_data * scale
        lidar_data += (width/2, height/2)
        
        # Invert Y coordinates for correct orientation
        lidar_data[:, 1] = height - lidar_data[:, 1] - 1
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.clip(lidar_data, 0, [width-1, height-1])
        
        # Create RGB image
        lidar_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Plot points with semantic colors
        for point_idx in range(len(lidar_data)):
            x, y = lidar_data[point_idx]
            tag = semantic_tags[point_idx]
            color = semantic_colors.get(tag, (255, 255, 255))  # White for unknown tags
            lidar_img[y, x] = color
        
        surface = pygame.surfarray.make_surface(lidar_img)
        # Rotate the surface 90° to the left
        return pygame.transform.rotate(surface, 90)
        
    except Exception as e:
        print(f"Error processing semantic lidar file {file_path.name}: {e}")
        return pygame.Surface(cell_size)

def process_imu(file_path, cell_size):
    """Process IMU data for visualization"""
    try:
        with open(file_path) as f:
            data = json.load(f)
        
        # Create a surface to display IMU data
        surface = pygame.Surface(cell_size)
        surface.fill((0, 0, 0))  # Black background
        
        # Calculate font size based on cell height
        font_size = min(32, cell_size[1] // 12)
        font = pygame.font.Font(None, font_size)
        title_font = pygame.font.Font(None, font_size + 8)
        
        # Prepare text lines without tuples
        lines = []
        lines.append(("", font, (255, 255, 255)))
        lines.append(("Accelerometer (m/s²)", title_font, (255, 255, 0)))
        lines.append((f"X: {data['accelerometer']['x']:8.3f}, Y: {data['accelerometer']['y']:8.3f}, Z: {data['accelerometer']['z']:8.3f}", font, (255, 255, 255)))
        lines.append(("", font, (255, 255, 255)))
        lines.append(("Gyroscope (rad/s)", title_font, (255, 255, 0)))
        lines.append((f"X: {data['gyroscope']['x']:8.3f}, Y: {data['gyroscope']['y']:8.3f}, Z: {data['gyroscope']['z']:8.3f}", font, (255, 255, 255)))
        lines.append(("", font, (255, 255, 255)))
        lines.append(("Compass", title_font, (255, 255, 0)))
        lines.append((f"{data['compass']:5.1f}°", font, (255, 255, 255)))
        
        # Calculate total height and starting position
        line_height = font_size + 4
        total_height = len(lines) * line_height
        start_y = (cell_size[1] - total_height) // 2
        
        # Render text
        for i, (text, font_obj, color) in enumerate(lines):
            if text:  # Only render non-empty lines
                text_surface = font_obj.render(text, True, color)
                text_rect = text_surface.get_rect(center=(cell_size[0]/2, start_y + i * line_height))
                surface.blit(text_surface, text_rect)
        
        return surface
        
    except Exception as e:
        print(f"Error processing IMU file {file_path}: {e}")
        return pygame.Surface(cell_size)

def process_gnss(file_path, cell_size):
    """Process GNSS data for visualization"""
    try:
        with open(file_path) as f:
            data = json.load(f)
        
        # Create a surface to display GNSS data
        surface = pygame.Surface(cell_size)
        surface.fill((0, 0, 0))  # Black background
        
        # Use same font sizes as IMU
        font_size = min(32, cell_size[1] // 12)
        font = pygame.font.Font(None, font_size)
        title_font = pygame.font.Font(None, font_size + 8)
        
        # Prepare text lines
        lines = []
        lines.append(("", font, (255, 255, 255)))
        lines.append(("Position", title_font, (255, 255, 0)))
        lines.append((f"Latitude:  {data['latitude']:11.6f}°", font, (255, 255, 255)))
        lines.append((f"Longitude: {data['longitude']:11.6f}°", font, (255, 255, 255)))
        lines.append((f"Altitude:  {data['altitude']:11.2f}m", font, (255, 255, 255)))
        
        # Calculate total height and starting position
        line_height = font_size + 4
        total_height = len(lines) * line_height
        start_y = (cell_size[1] - total_height) // 2
        
        # Render text
        for i, (text, font_obj, color) in enumerate(lines):
            if text:  # Only render non-empty lines
                text_surface = font_obj.render(text, True, color)
                text_rect = text_surface.get_rect(center=(cell_size[0]/2, start_y + i * line_height))
                surface.blit(text_surface, text_rect)
        
        return surface
        
    except Exception as e:
        print(f"Error processing GNSS file {file_path}: {e}")
        return pygame.Surface(cell_size)
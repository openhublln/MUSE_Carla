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
        img = pygame.image.load(str(file_path))
        surface = pygame.Surface(img.get_size(), pygame.SRCALPHA)
        surface.blit(img, (0, 0))

        # Annotation filename based on type (remains the same logic)
        if annotation_type == "3d":
            # This now points to the JSON file with the *new* structure
            bbox_file = str(file_path).replace('.png', '_3dbbox.json')
        else:
            # Keep 2d bbox logic if needed, although we focus on 3d clipping now
            bbox_file = str(file_path).replace('.png', '_bbox.json')

        if os.path.exists(bbox_file):
            try:
                with open(bbox_file, 'r') as f:
                    annotation_data = json.load(f) # Load the list of actor data

                # --- NEW DRAWING LOGIC for Clipped Segments ---
                if annotation_type == "3d":
                    line_color = (0, 255, 0) # Green for clipped edges
                    line_thickness = 2

                    # Check if the loaded data is a list (new format)
                    if isinstance(annotation_data, list):
                        for actor_data in annotation_data:
                            if 'clipped_segments' in actor_data:
                                for segment in actor_data['clipped_segments']:
                                    # segment is [[x1, y1], [x2, y2]]
                                    p1 = (int(segment[0][0]), int(segment[0][1]))
                                    p2 = (int(segment[1][0]), int(segment[1][1]))
                                    pygame.draw.line(surface, line_color, p1, p2, line_thickness)

                            # Optional: Draw the bbox derived from clipped points
                            # if 'bbox_from_clipped' in actor_data and actor_data['bbox_from_clipped']:
                            #     x, y, w, h = actor_data['bbox_from_clipped']
                            #     pygame.draw.rect(surface, (255, 165, 0), (x, y, w, h), 1) # Orange outline

                    # --- Fallback/Compatibility for OLD 3D format (8 vertices) ---
                    elif isinstance(annotation_data, dict) and 'bounding_boxes' in annotation_data:
                         # Handle old format if necessary, or remove this block
                         # This block is less relevant now but kept for potential compatibility
                         boxes = annotation_data.get('bounding_boxes', [])
                         edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
                         for bbox in boxes:
                             if 'vertices' in bbox:
                                 verts = bbox['vertices']
                                 # Draw lines using old vertex data (might be distorted)
                                 # Add checks for INVALID_POINT if using the intermediate solution
                                 valid_verts = [v for v in verts if v != [-1.0, -1.0]] # Example check
                                 if len(valid_verts) > 1: # Need at least 2 points to draw edges
                                     try:
                                         for edge in edges:
                                            # Basic check if indices are valid and points are not marked invalid
                                            if max(edge) < len(verts) and verts[edge[0]] != [-1.0, -1.0] and verts[edge[1]] != [-1.0, -1.0]:
                                                start = verts[edge[0]]
                                                end = verts[edge[1]]
                                                pygame.draw.line(surface, (255, 0, 0), # Red for old/potentially distorted
                                                                    (int(start[0]), int(start[1])),
                                                                    (int(end[0]), int(end[1])), 2)
                                     except Exception as draw_err:
                                        print(f"Minor error drawing old format edge: {draw_err}")


                # --- Existing 2D Annotation Drawing ---
                else: # annotation_type == "2d"
                     # Check standard 2D bbox format
                    if isinstance(annotation_data, dict) and 'bounding_boxes' in annotation_data:
                        boxes = annotation_data.get('bounding_boxes', [])
                        for bbox in boxes:
                             if 'bbox' in bbox: # Standard 2d [x, y, w, h]
                                x, y, w, h = bbox['bbox']
                                pygame.draw.rect(surface, (255, 0, 0), (x, y, w, h), 2)
                                # Add distance text if available
                                if 'distance' in bbox:
                                     font = pygame.font.Font(None, 24)
                                     dist_text = font.render(f"{bbox['distance']:.1f}m", True, (255,255,255))
                                     surface.blit(dist_text, (x, y-20))

                return surface # Return the surface with drawings

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file {bbox_file}: {e}")
                return img # Return original image on JSON error
            except Exception as e:
                print(f"Error processing annotations file {bbox_file}: {e}")
                return img # Return original image on other processing errors

        return img # Return original image if no annotation file found

    except pygame.error as e:
        print(f"Error loading image file {file_path}: {e}")
        # Return a blank surface matching cell_size if image loading fails
        return pygame.Surface(cell_size, pygame.SRCALPHA)
    except Exception as e:
        print(f"General error processing camera {file_path.name}: {e}")
        # Return a blank surface matching cell_size on other errors
        return pygame.Surface(cell_size, pygame.SRCALPHA)

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
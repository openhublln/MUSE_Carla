import os
import sys
import numpy as np
import pygame
import time
from pathlib import Path
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import yaml
import json

class FlexibleDataPlayer:
    """ Flexible player to support various sensors (camera, radar, lidar) """
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise RuntimeError(f"Data directory does not exist: {self.data_dir}")
        print(f"Initializing player with data directory: {self.data_dir}")

        # Charger la config pour récupérer la liste des capteurs
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)
        sensors_config = config["sensors"]
        
        # Construire la liste des capteurs à partir de la config
        self.sensors = {}
        for sensor_cfg in sensors_config:
            sname = sensor_cfg["name"]
            stype = sensor_cfg["type"].lower()
            # Update pattern for IMU files
            pattern = "*.json" if stype in ["gnss", "imu"] else "*.png" if stype == "camera" else "*.npy"
            sensor_folder = self.data_dir / sname
            if sensor_folder.exists() and sensor_folder.is_dir():
                files = sorted(list(sensor_folder.glob(pattern)))
                if files:
                    self.sensors[sname] = {
                        "type": stype,
                        "path": sensor_folder,
                        "files": files,
                        "last_valid": None,
                        "name": sname  # Ajout du nom du capteur dans le dictionnaire
                    }
        if not self.sensors:
            raise RuntimeError("No sensor data found using configuration!")
        print(f"Detected sensors: {list(self.sensors.keys())}")
        
        # Compute common timestamps by finding intersection of file stems (as int)
        ts_sets = []
        for sensor in self.sensors.values():
            ts = set(int(f.stem) for f in sensor["files"])
            ts_sets.append(ts)
        self.timestamps = sorted(list(set.intersection(*ts_sets))) if ts_sets else []
        if not self.timestamps:
            raise RuntimeError("No common timestamps found among sensors!")
        print(f"Found {len(self.timestamps)} common timestamps.")

        # Compute grid size
        self.num_sensors = len(self.sensors)
        cols = int(math.ceil(math.sqrt(self.num_sensors)))
        rows = int(math.ceil(self.num_sensors / cols))
        self.grid = (rows, cols)
        # Base cell size 
        base_cell_width, base_cell_height = 800, 600
        
        pygame.init()
        # Get display info for dynamic scaling
        info = pygame.display.Info()
        screen_width, screen_height = info.current_w, info.current_h
        desired_width = cols * base_cell_width
        desired_height = rows * base_cell_height
        # Limit window height to 75% of screen height
        scale_factor = min(1, screen_width/desired_width, (screen_height * 0.75)/desired_height)
        cell_width = int(base_cell_width * scale_factor)
        cell_height = int(base_cell_height * scale_factor)
        self.cell_size = (cell_width, cell_height)
        self.window_size = (cols * cell_width, rows * cell_height)
        
        self.display = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Flexible Sensor Replay")
        self.clock = pygame.time.Clock()

        self.current_index = 0
        self.auto_play = True
        self.frame_time = 0.05
        self.last_update = time.time()

        # Update semantic color mapping according to CityScapesPalette.h
        self.semantic_colors = {
            0: (0, 0, 0),         # unlabeled
            1: (128, 64, 128),    # road
            2: (244, 35, 232),    # sidewalk
            3: (70, 70, 70),      # building
            4: (102, 102, 156),   # wall
            5: (190, 153, 153),   # fence
            6: (153, 153, 153),   # pole
            7: (250, 170, 30),    # traffic light
            8: (220, 220, 0),     # traffic sign
            9: (107, 142, 35),    # vegetation
            10: (152, 251, 152),  # terrain
            11: (70, 130, 180),   # sky
            12: (220, 20, 60),    # pedestrian
            13: (255, 0, 0),      # rider
            14: (0, 0, 142),      # Car
            15: (0, 0, 70),       # truck
            16: (0, 60, 100),     # bus
            17: (0, 80, 100),     # train
            18: (0, 0, 230),      # motorcycle
            19: (119, 11, 32),    # bicycle
            20: (110, 190, 160),  # static
            21: (170, 120, 50),   # dynamic
            22: (55, 90, 80),     # other
            23: (45, 60, 150),    # water
            24: (157, 234, 50),   # road line
            25: (81, 0, 81),      # ground
            26: (150, 100, 100),  # bridge
            27: (230, 150, 140),  # rail track
            28: (180, 165, 180),  # guard rail
            29: (180, 130, 70),   # rock
        }

    def process_camera(self, file_path, sensor_name):
        """Process camera image and overlay bounding boxes if available"""
        try:
            # Charger l'image
            img = pygame.image.load(str(file_path))
            
            # Chercher le fichier bbox correspondant
            bbox_file = str(file_path).replace('.png', '_bbox.json')
            if os.path.exists(bbox_file):
                try:
                    with open(bbox_file, 'r') as f:
                        bbox_data = json.load(f)
                        
                    # Créer une surface pygame à partir de l'image
                    surface = pygame.Surface(img.get_size(), pygame.SRCALPHA)
                    surface.blit(img, (0,0))
                    
                    # Dessiner les bounding boxes
                    for bbox in bbox_data['bounding_boxes']:
                        x, y, w, h = bbox['bbox']
                        # Dessiner uniquement le contour du rectangle en rouge
                        pygame.draw.rect(surface, (255, 0, 0), (x, y, w, h), 2)
                        
                        # Afficher la distance si disponible
                        if 'distance' in bbox:
                            font = pygame.font.Font(None, 24)
                            dist_text = font.render(f"{bbox['distance']:.1f}m", True, (255, 255, 255))
                            surface.blit(dist_text, (x, y-20))
                            
                    return surface
                except Exception as e:
                    print(f"Error loading bbox data: {e}")
                    return img
            return img
        except Exception as e:
            print(f"Error loading camera image: {e}")
            return pygame.Surface(self.cell_size)

    def scale_to_fit(self, surface, target_size):
        """Redimensionne la surface pour tenir dans target_size tout en préservant l'aspect ratio."""
        sw, sh = surface.get_size()
        tw, th = target_size
        scale = min(tw / sw, th / sh)
        new_size = (max(1, int(sw * scale)), max(1, int(sh * scale)))
        return pygame.transform.smoothscale(surface, new_size)

    def process_radar(self, file_path):
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
            return pygame.Surface(self.cell_size)

    def process_lidar(self, file_path):
        try:
            points = np.load(file_path)
            # Create a blank image for display
            width, height = self.cell_size
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
            return pygame.Surface(self.cell_size)

    def process_semantic_lidar(self, file_path):
        try:
            # Load the semantic LIDAR data
            points = np.load(file_path)
            
            # Create a blank image for display
            width, height = self.cell_size
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
                color = self.semantic_colors.get(tag, (255, 255, 255))  # White for unknown tags
                lidar_img[y, x] = color
            
            surface = pygame.surfarray.make_surface(lidar_img)
            # Rotate the surface 90° to the left
            return pygame.transform.rotate(surface, 90)
            
        except Exception as e:
            print(f"Error processing semantic lidar file {file_path.name}: {e}")
            return pygame.Surface(self.cell_size)

    def process_imu(self, file_path):
        """Process IMU data for visualization"""
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            # Create a surface to display IMU data
            surface = pygame.Surface(self.cell_size)
            surface.fill((0, 0, 0))  # Black background
            
            # Render IMU data
            font = pygame.font.Font(None, 32)
            texts = [
                "Accelerometer (m/s²):",
                f"X: {data['accelerometer']['x']:.2f}",
                f"Y: {data['accelerometer']['y']:.2f}",
                f"Z: {data['accelerometer']['z']:.2f}",
                "",
                "Gyroscope (rad/s):",
                f"X: {data['gyroscope']['x']:.2f}",
                f"Y: {data['gyroscope']['y']:.2f}",
                f"Z: {data['gyroscope']['z']:.2f}",
                "",
                f"Compass: {data['compass']:.2f}°"
            ]
            
            y_offset = 20
            for text in texts:
                text_surface = font.render(text, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(self.cell_size[0]/2, y_offset))
                surface.blit(text_surface, text_rect)
                y_offset += 30
                
            return surface
            
        except Exception as e:
            print(f"Error processing IMU file {file_path}: {e}")
            return pygame.Surface(self.cell_size)

    def process_sensor(self, sensor, timestamp):
        # Look for a file whose stem equals the timestamp
        file = next((f for f in sensor["files"] if int(f.stem) == timestamp), None)
        if file:
            if sensor["type"] == "imu":
                return self.process_imu(file)
            elif sensor["type"] == "gnss":
                return self.process_gnss(file)
            elif sensor["type"] == "camera":
                processed = self.process_camera(file, sensor["name"])  # Le nom est maintenant accessible
            elif sensor["type"] == "radar":
                processed = self.process_radar(file)
            elif sensor["type"] == "lidar":
                if "semantic" in sensor["name"]:
                    processed = self.process_semantic_lidar(file)
                else:
                    processed = self.process_lidar(file)
            sensor["last_valid"] = processed
            return processed
        else:
            # Use last valid if available
            return sensor["last_valid"] if sensor["last_valid"] is not None else pygame.Surface(self.cell_size)

    def process_gnss(self, file_path):
        """Process GNSS data for visualization"""
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            # Create a surface to display GNSS data
            surface = pygame.Surface(self.cell_size)
            surface.fill((0, 0, 0))  # Black background
            
            # Render text
            font = pygame.font.Font(None, 36)
            texts = [
                f"Latitude: {data['latitude']:.6f}°",
                f"Longitude: {data['longitude']:.6f}°",
                f"Altitude: {data['altitude']:.2f}m"
            ]
            
            y_offset = 20
            for text in texts:
                text_surface = font.render(text, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(self.cell_size[0]/2, y_offset))
                surface.blit(text_surface, text_rect)
                y_offset += 40
                
            return surface
            
        except Exception as e:
            print(f"Error processing GNSS file {file_path}: {e}")
            return pygame.Surface(self.cell_size)

    def run(self):
        running = True
        sensor_keys = list(self.sensors.keys())
        rows, cols = self.grid
        window_width, window_height = self.window_size
        
        # Définir des marges pour le header et le footer
        header_height = 60
        footer_height = 60
        
        # Calculer les dimensions effectives de la zone de grille
        effective_height = window_height - header_height - footer_height
        effective_cell_height = effective_height / rows
        effective_cell_width = window_width / cols
        
        # Initialisation des polices pour afficher le titre, timestamp et les noms de capteurs
        sensor_font = pygame.font.Font(None, 24)
        scene_font = pygame.font.Font(None, 36)
        
        # Extraire le nom de la scène à partir du chemin de données
        scene_name = os.path.basename(os.path.normpath(str(self.data_dir)))
        while running:
            current_time = time.time()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_LEFT,):
                        self.auto_play = False
                        self.current_index = max(0, self.current_index - 1)
                    elif event.key in (pygame.K_RIGHT,):
                        self.auto_play = False
                        self.current_index = min(len(self.timestamps) - 1, self.current_index + 1)
                    elif event.key == pygame.K_SPACE:
                        self.auto_play = not self.auto_play
                        print(f"\nAuto-play: {'ON' if self.auto_play else 'OFF'}")
    
            if self.auto_play and (current_time - self.last_update) >= self.frame_time:
                if self.current_index < len(self.timestamps) - 1:
                    self.current_index += 1
                    self.last_update = current_time
                else:
                    self.auto_play = False
                    print("\nReached end of sequence. Auto-play deactivated.")
    
            # Clear display et dessiner les zones réservées au header et footer
            self.display.fill((0,0,0))
            pygame.draw.rect(self.display, (0,0,0), (0, 0, window_width, header_height))
            pygame.draw.rect(self.display, (0,0,0), (0, window_height - footer_height, window_width, footer_height))
            
            ts = self.timestamps[self.current_index]
            # Afficher le titre de la scène au centre du header
            scene_text = scene_font.render(f"Scene: {scene_name}", True, (255,255,255))
            scene_rect = scene_text.get_rect(center=(window_width//2, header_height//2))
            self.display.blit(scene_text, scene_rect)
            
            for idx, key in enumerate(sensor_keys):
                sensor = self.sensors[key]
                img = self.process_sensor(sensor, ts)
                r = idx // cols
                c = idx % cols
                # Si la dernière ligne contient un seul capteur
                if r == (rows - 1) and (len(sensor_keys) % cols == 1):
                    target = (window_width, effective_cell_height)
                    cell_x, cell_y = 0, header_height + r * effective_cell_height
                else:
                    target = (effective_cell_width, effective_cell_height)
                    cell_x, cell_y = c * effective_cell_width, header_height + r * effective_cell_height
                # Redimensionner l'image sans déformer (aspect ratio conservé)
                scaled_img = self.scale_to_fit(img, target)
                offset_x = (target[0] - scaled_img.get_width()) // 2
                offset_y = (target[1] - scaled_img.get_height()) // 2
                self.display.blit(scaled_img, (cell_x + offset_x, cell_y + offset_y))
                # Afficher le nom du capteur en haut à gauche de la cellule
                sensor_text = sensor_font.render(key, True, (255,255,255))
                self.display.blit(sensor_text, (cell_x + 5, cell_y + 5))
            
            # Afficher le timestamp dans le footer centré horizontalement
            timestamp_text = scene_font.render(f"Timestamp: {ts} ({self.current_index+1}/{len(self.timestamps)})", True, (255,255,255))
            timestamp_rect = timestamp_text.get_rect(center=(window_width//2, window_height - footer_height//2))
            self.display.blit(timestamp_text, timestamp_rect)
            pygame.display.flip()
            self.clock.tick(20)
        pygame.quit()

if __name__ == "__main__":
    try:
        # Charger la configuration pour obtenir le chemin de base des scènes
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)
        base_save_path = config["simulation"]["base_save_path"]
        
        # Choisir manuellement la scène à visualiser (via ligne de commande ou valeur par défaut)
        if len(sys.argv) > 1:
            scene_name = sys.argv[1]
        else:
            scene_name = "scene_1"  # Valeur par défaut, à modifier selon vos besoins
        
        data_dir = os.path.join(base_save_path, scene_name)
        print(f"Starting flexible replay from: {data_dir}")
        player = FlexibleDataPlayer(data_dir)
        player.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)
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
from replay_processing import (
    process_camera,
    process_radar,
    process_lidar,
    process_semantic_lidar,
    process_imu,
    process_gnss
)

class FlexibleDataPlayer:
    """ Flexible player to support various sensors (camera, radar, lidar) """
    def __init__(self, data_dir, annotation_type="2d"):
        self.data_dir = Path(data_dir)
        self.annotation_type = annotation_type  # new parameter for annotation type ("2d" or "3d")
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
            stype = sensor_cfg["type"].lower().replace(" ", "_")
            blueprint = sensor_cfg["blueprint"].lower()  # Normalize blueprint string
            
            # Define pattern and actual type based on blueprint
            if "semantic_segmentation" in blueprint:
                pattern = "*.png"
                actual_type = "semantic_camera"
            elif "instance_segmentation" in blueprint:
                pattern = "*.png"
                actual_type = "instance_camera"
            elif "camera" in blueprint and "rgb" in blueprint:
                pattern = "*.png"
                actual_type = "camera"
            elif "semantic" in blueprint and "lidar" in blueprint:
                pattern = "*.npy"
                actual_type = "semantic_lidar"
            elif "radar" in blueprint:
                pattern = "*.npy"
                actual_type = "radar"
            elif "lidar" in blueprint:
                pattern = "*.npy"
                actual_type = "lidar"
            else:
                pattern = "*.json"
                actual_type = stype

            sensor_folder = self.data_dir / sname
            if sensor_folder.exists() and sensor_folder.is_dir():
                files = sorted(list(sensor_folder.glob(pattern)))
                if files:
                    self.sensors[sname] = {
                        "type": actual_type,
                        "path": sensor_folder,
                        "files": files,
                        "last_valid": None,
                        "name": sname
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

    def scale_to_fit(self, surface, target_size):
        """Redimensionne la surface pour tenir dans target_size tout en préservant l'aspect ratio."""
        sw, sh = surface.get_size()
        tw, th = target_size
        scale = min(tw / sw, th / sh)
        new_size = (max(1, int(sw * scale)), max(1, int(sh * scale)))
        return pygame.transform.smoothscale(surface, new_size)

    def process_sensor(self, sensor, timestamp):
        """Process sensor data for visualization"""
        file = next((f for f in sensor["files"] if int(f.stem) == timestamp), None)
        
        if file:
            try:
                sensor_type = sensor["type"]
                if sensor_type == "imu":
                    return process_imu(file, self.cell_size)
                elif sensor_type == "gnss":
                    return process_gnss(file, self.cell_size)
                elif sensor_type == "camera":
                    return process_camera(file, sensor["name"], self.annotation_type, self.cell_size)
                elif sensor_type == "semantic_camera":
                    return pygame.image.load(str(file))  # Load semantic segmentation directly
                elif sensor_type == "instance_camera":
                    return pygame.image.load(str(file))  # Load instance segmentation directly
                elif sensor_type == "radar":
                    return process_radar(file, self.cell_size)
                elif sensor_type == "semantic_lidar":
                    return process_semantic_lidar(file, self.cell_size, self.semantic_colors)
                elif sensor_type == "lidar":
                    return process_lidar(file, self.cell_size)
                
            except Exception as e:
                print(f"Error processing {sensor['name']} at timestamp {timestamp}: {e}")
        
        return sensor["last_valid"] if sensor["last_valid"] is not None else pygame.Surface(self.cell_size)

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
        
        # Choisir manuellement la scène à visualiser
        if len(sys.argv) > 1:
            scene_name = sys.argv[1]
        else:
            scene_name = "scene_1"  # Valeur par défaut
        
        # New: Read annotation type ("2d" or "3d") if provided (default "2d")
        if len(sys.argv) > 2:
            annotation_type = sys.argv[2]
        else:
            annotation_type = "2d"
            
        data_dir = os.path.join(base_save_path, scene_name)
        print(f"Starting flexible replay from: {data_dir} with '{annotation_type}' annotations")
        player = FlexibleDataPlayer(data_dir, annotation_type)
        player.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)
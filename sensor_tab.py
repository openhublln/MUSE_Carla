from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QComboBox, QPushButton, QScrollArea)
from PyQt6.QtCore import pyqtSignal
from sensor_widgets import SensorWidget

class SensorTab(QWidget):
    """Tab for sensor configuration"""
    configChanged = pyqtSignal()
    
    # Add "Custom Sensor" to the beginning of the preset order
    SENSOR_PRESET_ORDER = ["Custom Sensor"] + [
        "Camera_Front", "Camera_Back", "Camera_FrontRight", "Camera_FrontLeft", 
        "Camera_BackRight", "Camera_BackLeft", "Radar_Front", "Radar_FrontRight", 
        "Radar_FrontLeft", "Radar_BackRight", "Radar_BackLeft", "Lidar", 
        "Semantic_Lidar", "GNSS", "IMU"
    ]
    
    # Add a custom sensor preset with default values
    SENSOR_PRESETS = {
        "Custom Sensor": {
            "type": "Camera",
            "attributes": {
                "image_size_x": 800,
                "image_size_y": 600,
                "fov": 90.0,
            },
            "transform": {
                "location": {"x": 0, "y": 0, "z": 2.4},
                "rotation": {"yaw": 0}
            },
            "collect_bbox": True
        },
        "Camera_Front": {
            "type": "Camera",
            "attributes": {
                "image_size_x": 800,
                "image_size_y": 600,
                "fov": 90.0,
            },
            "transform": {
                "location": {"x": 0.4, "y": 0, "z": 2.4},
                "rotation": {"yaw": 0}
            },
            "collect_bbox": True
        },
        "Camera_Back": {
            "type": "Camera",
            "attributes": {
                "image_size_x": 800,
                "image_size_y": 600,
                "fov": 90.0,
            },
            "transform": {
                "location": {"x": -1, "y": 0, "z": 2.4},
                "rotation": {"yaw": 180}
            },
            "collect_bbox": True
        },
        "Camera_FrontRight": {
            "type": "Camera",
            "attributes": {
                "image_size_x": 800,
                "image_size_y": 600,
                "fov": 90.0,
            },
            "transform": {
                "location": {"x": 0.4, "y": 0.3, "z": 2.4},
                "rotation": {"yaw": 45}
            },
            "collect_bbox": True
        },
        "Camera_FrontLeft": {
            "type": "Camera",
            "attributes": {
                "image_size_x": 800,
                "image_size_y": 600,
                "fov": 90.0,
            },
            "transform": {
                "location": {"x": 0.4, "y": -0.3, "z": 2.4},
                "rotation": {"yaw": -45}
            },
            "collect_bbox": True
        },
        "Camera_BackRight": {
            "type": "Camera",
            "attributes": {
                "image_size_x": 800,
                "image_size_y": 600,
                "fov": 90.0,
            },
            "transform": {
                "location": {"x": -0.3, "y": 0.5, "z": 2.4},
                "rotation": {"yaw": -225} 
            },
            "collect_bbox": True
        },
        "Camera_BackLeft": {
            "type": "Camera",
            "attributes": {
                "image_size_x": 800,
                "image_size_y": 600,
                "fov": 90.0,
            },
            "transform": {
                "location": {"x": -0.3, "y": -0.5, "z": 2.4},
                "rotation": {"yaw": 225}  
            },
            "collect_bbox": True
        },
        "Radar_Front": {
            "type": "Radar",
            "attributes": {
                "horizontal_fov": 90,
                "vertical_fov": 10,
                "points_per_second": 5000,
                "range": 250
            },
            "transform": {
                "location": {"x": 2.5, "y": 0, "z": 1},
                "rotation": {"pitch": 5, "yaw": 0, "roll": 0}
            }
        },
        "Radar_FrontRight": {
            "type": "Radar",
            "attributes": {
                "horizontal_fov": 90,
                "vertical_fov": 10,
                "points_per_second": 5000,
                "range": 250
            },
            "transform": {
                "location": {"x": 1.5, "y": 1, "z": 1},
                "rotation": {"pitch": 5, "yaw": 90, "roll": 0}
            }
        },
        "Radar_FrontLeft": {
            "type": "Radar",
            "attributes": {
                "horizontal_fov": 90,
                "vertical_fov": 10,
                "points_per_second": 5000,
                "range": 250
            },
            "transform": {
                "location": {"x": 1.5, "y": -1, "z": 1},
                "rotation": {"pitch": 5, "yaw": -90, "roll": 0}
            }
        },
        "Radar_BackRight": {
            "type": "Radar",
            "attributes": {
                "horizontal_fov": 90,
                "vertical_fov": 10,
                "points_per_second": 5000,
                "range": 250
            },
            "transform": {
                "location": {"x": -2.5, "y": 0.5, "z": 1},
                "rotation": {"pitch": 5, "yaw": 180, "roll": 0}
            }
        },        
        "Radar_BackLeft": {
            "type": "Radar",
            "attributes": {
                "horizontal_fov": 90,
                "vertical_fov": 10,
                "points_per_second": 5000,
                "range": 250
            },
            "transform": {
                "location": {"x": -2.5, "y": -0.5, "z": 1},
                "rotation": {"pitch": 5, "yaw": 180, "roll": 0}
            }
        },
        "Lidar": {
            "type": "LIDAR",
            "blueprint": "sensor.lidar.ray_cast",
            "attributes": {
                "channels": 64,
                "range": 100,
                "points_per_second": 250000,
                "rotation_frequency": 20.0,
                "upper_fov": 10.0,
                "lower_fov": -30.0,
                "horizontal_fov": 360.0,
                "atmosphere_attenuation_rate": 0.004,
                "dropoff_general_rate": 0.45,
                "dropoff_intensity_limit": 0.8,
                "dropoff_zero_intensity": 0.4,
                "noise_stddev": 0.0
            },
            "transform": {
                "location": {"x": -0.3, "y": 0, "z": 2.4},
                "rotation": {"yaw": 90}
            }
        },
        "Semantic_Lidar": {
            "type": "Semantic LIDAR",
            "attributes": {
                "channels": 64,
                "range": 100,
                "points_per_second": 250000,
                "rotation_frequency": 20,
                "upper_fov": 10,
                "lower_fov": -30,
                "horizontal_fov": 360
            },
            "transform": {
                "location": {"x": -0.3, "y": 0, "z": 2.4},
                "rotation": {"yaw": 90}
            }
        },
        "GNSS": {
            "type": "GNSS",
            "attributes": {
                "noise_alt_bias": 0.0,
                "noise_alt_stddev": 0.1,
                "noise_lat_bias": 0.0,
                "noise_lat_stddev": 0.1,
                "noise_lon_bias": 0.0,
                "noise_lon_stddev": 0.1
            },
            "transform": {
                "location": {"x": -1, "y": 0, "z": 2.4},
                "rotation": {"pitch": 0, "yaw": 0, "roll": 0}
            }
        },
        "IMU": {
            "type": "IMU",
            "attributes": {
                "noise_accel_stddev_x": 0.1,
                "noise_accel_stddev_y": 0.1,
                "noise_accel_stddev_z": 0.1,
                "noise_gyro_stddev_x": 0.1,
                "noise_gyro_stddev_y": 0.1,
                "noise_gyro_stddev_z": 0.1,
                "noise_gyro_bias_x": 0.0,
                "noise_gyro_bias_y": 0.0,
                "noise_gyro_bias_z": 0.0
            },
            "transform": {
                "location": {"x": -1, "y": 0, "z": 2.4},
                "rotation": {"pitch": 0, "yaw": 0, "roll": 0}
            }
        }
    }

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        
        # Create button container with simpler layout
        button_container = QWidget()
        button_layout = QHBoxLayout()
        
        # Add preset selector dropdown
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(self.SENSOR_PRESET_ORDER)
        button_layout.addWidget(QLabel("Select Sensor :"))
        button_layout.addWidget(self.preset_combo)
        
        # Single "Add Sensor" button
        add_btn = QPushButton("Add Sensor")
        add_btn.clicked.connect(self._add_preset)  
        button_layout.addWidget(add_btn)
        
        button_container.setLayout(button_layout)
        self.layout.addWidget(button_container)
        
        # Scroll area for sensors
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.sensors_widget = QWidget()
        self.sensors_layout = QVBoxLayout()
        self.sensors_widget.setLayout(self.sensors_layout)
        scroll.setWidget(self.sensors_widget)
        self.layout.addWidget(scroll)
        
        self.setLayout(self.layout)
        self.sensors = []
    
    def _add_sensor(self):
        # This method is no longer needed but kept for compatibility
        self._add_preset()
    
    def _add_preset(self):
        """Add either a custom or pre-configured sensor based on selection"""
        preset_name = self.preset_combo.currentText()
        preset = self.SENSOR_PRESETS[preset_name]
        
        sensor = SensorWidget(self)
        sensor.configChanged.connect(self.configChanged.emit)
        sensor.deleteRequested.connect(self._remove_sensor)
        
        # For custom sensor, just set a unique name
        if preset_name == "Custom Sensor":
            sensor.name.setText(f"new_sensor_{len(self.sensors)}")
        else:
            # Configure the sensor according to preset
            sensor.name.setText(preset_name)
            sensor.type.setCurrentText(preset["type"])
            
            # Set attributes
            for name, value in preset["attributes"].items():
                if name in sensor.attributes_dict:
                    sensor.attributes_dict[name].setValue(value)
            
            # Set transform with full rotation values
            if "transform" in preset:
                if "location" in preset["transform"]:
                    loc = preset["transform"]["location"]
                    sensor.transform.location.x.setValue(loc.get("x", 0))
                    sensor.transform.location.y.setValue(loc.get("y", 0))
                    sensor.transform.location.z.setValue(loc.get("z", 0))
                
                if "rotation" in preset["transform"]:
                    rot = preset["transform"]["rotation"]
                    # Only set the provided rotation values
                    if "pitch" in rot:
                        sensor.transform.rotation.pitch.setValue(float(rot["pitch"]))
                    if "yaw" in rot:
                        sensor.transform.rotation.yaw.setValue(float(rot["yaw"]))
                    if "roll" in rot:
                        sensor.transform.rotation.roll.setValue(float(rot["roll"]))
            
            # Set bbox collection for cameras
            if preset["type"] == "Camera" and "collect_bbox" in preset:
                sensor.collect_bbox.setChecked(preset["collect_bbox"])
        
        self.sensors.append(sensor)
        self.sensors_layout.addWidget(sensor)
        self.configChanged.emit()
    
    def _remove_sensor(self, sensor):
        self.sensors.remove(sensor)
        sensor.deleteLater()
        self.configChanged.emit()
    
    def get_config(self):
        """Return the list of sensor configurations"""
        return [sensor.get_config() for sensor in self.sensors]

import sys
import yaml
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTabWidget, QLabel, QLineEdit, QSpinBox, 
                            QDoubleSpinBox, QCheckBox, QPushButton, QComboBox,
                            QScrollArea, QFileDialog, QMessageBox, QGroupBox,
                            QTextEdit, QSplitter, QDialog)  
from PyQt6.QtCore import Qt, pyqtSignal, QLocale
import os
import subprocess
from pathlib import Path  

class SimulationTab(QWidget):
    """Tab for simulation settings"""
    configChanged = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        
        # Simulation Group
        sim_group = QGroupBox("Simulation Settings")
        sim_layout = QVBoxLayout()
        
        # Basic settings
        num_scenes_container = self._create_spinbox("Number of Scenes:", 1, 1000, 1)
        self.num_scenes = num_scenes_container.findChild(QSpinBox)
        
        ticks_container = self._create_spinbox("Ticks per Scene:", 1, 1000, 20)
        self.ticks_per_scene = ticks_container.findChild(QSpinBox)
        
        self.base_save_path = self._create_path_selector("Base Save Path:", "./_out")
        
        # Traffic Group
        traffic_group = QGroupBox("Traffic Settings")
        traffic_layout = QVBoxLayout()
        
        vehicles_container = self._create_spinbox("Number of Vehicles:", 0, 100, 30)
        self.num_vehicles = vehicles_container.findChild(QSpinBox)
        
        pedestrians_container = self._create_spinbox("Number of Pedestrians:", 0, 100, 10)
        self.num_pedestrians = pedestrians_container.findChild(QSpinBox)
        
        self.safe_spawn = QCheckBox("Safe Spawn")
        self.safe_spawn.setChecked(True)
        self.car_lights_on = QCheckBox("Car Lights On")
        self.car_lights_on.setChecked(True)
        
        for widget in [vehicles_container, pedestrians_container, 
                      self.safe_spawn, self.car_lights_on]:
            traffic_layout.addWidget(widget)
        traffic_group.setLayout(traffic_layout)
        
        # Add all widgets to main layout
        for widget in [num_scenes_container, ticks_container, 
                      self.base_save_path, traffic_group]:
            sim_layout.addWidget(widget)
        
        sim_group.setLayout(sim_layout)
        self.layout.addWidget(sim_group)
        self.layout.addStretch()
        self.setLayout(self.layout)
        
        # Connect signals
        for spinbox in [self.num_scenes, self.ticks_per_scene, 
                       self.num_vehicles, self.num_pedestrians]:
            spinbox.valueChanged.connect(self.configChanged.emit)
        for checkbox in [self.safe_spawn, self.car_lights_on]:
            checkbox.stateChanged.connect(self.configChanged.emit)
    
    def _create_spinbox(self, label, min_val, max_val, default):
        container = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default)
        layout.addWidget(spinbox)
        container.setLayout(layout)
        return container
    
    def _create_path_selector(self, label, default):
        container = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        self.path_edit = QLineEdit(default)
        layout.addWidget(self.path_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_path)
        layout.addWidget(browse_btn)
        container.setLayout(layout)
        return container
    
    def _browse_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.path_edit.setText(path)
            self.configChanged.emit()
    
    def get_config(self):
        """Return the current configuration as a dictionary"""
        return {
            "simulation": {
                "num_scenes": self.num_scenes.value(),
                "ticks_per_scene": self.ticks_per_scene.value(),
                "base_save_path": self.path_edit.text(),
                "traffic": {
                    "num_vehicles": self.num_vehicles.value(),
                    "num_pedestrians": self.num_pedestrians.value(),
                    "safe_spawn": self.safe_spawn.isChecked(),
                    "car_lights_on": self.car_lights_on.isChecked(),
                    "seed": None
                }
            }
        }

class SensorTab(QWidget):
    """Tab for sensor configuration"""
    configChanged = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        
        # Add sensor button
        add_btn = QPushButton("Add Sensor")
        add_btn.clicked.connect(self._add_sensor)
        self.layout.addWidget(add_btn)
        
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
        sensor = SensorWidget(self)
        sensor.configChanged.connect(self.configChanged.emit)
        sensor.deleteRequested.connect(self._remove_sensor)
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

class TransformWidget(QGroupBox):
    """Widget for transform configuration (combines location and rotation)"""
    configChanged = pyqtSignal()  # Add signal
    
    def __init__(self):
        super().__init__("Transform")
        layout = QVBoxLayout()
        
        self.location = LocationWidget()
        self.rotation = RotationWidget()
        
        # Connect location and rotation signals to configChanged
        self.location.valueChanged.connect(self.configChanged.emit)
        self.rotation.valueChanged.connect(self.configChanged.emit)
        
        layout.addWidget(self.location)
        layout.addWidget(self.rotation)
        self.setLayout(layout)
    
    def get_config(self):
        return {
            "location": self.location.get_config(),
            "rotation": self.rotation.get_config()
        }

class SensorWidget(QGroupBox):
    """Widget for individual sensor configuration"""
    configChanged = pyqtSignal()
    deleteRequested = pyqtSignal(object)
    
    SENSOR_TYPES = {
        "Camera": "sensor.camera.rgb",
        "Radar": "sensor.other.radar",
        "LIDAR": "sensor.lidar.ray_cast",
        "Semantic LIDAR": "sensor.lidar.ray_cast_semantic",
        "GNSS": "sensor.other.gnss",
        "IMU": "sensor.other.imu"
    }
    
    def __init__(self, parent=None):
        super().__init__("Sensor Configuration")
        self.layout = QVBoxLayout()
        self.attributes_dict = {}  # Initialize attributes dict
        
        # Name field
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name = QLineEdit("new_sensor")
        self.name.textChanged.connect(self.configChanged.emit)
        name_layout.addWidget(self.name)
        
        # Type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type = QComboBox()
        self.type.addItems(self.SENSOR_TYPES.keys())
        self.type.currentTextChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.type)
        
        # Bbox annotation widget for cameras
        self.bbox_widget = QWidget()
        bbox_layout = QHBoxLayout()
        self.collect_bbox = QCheckBox("Enable Bounding Box Collection")
        self.collect_bbox.setChecked(True)
        self.collect_bbox.stateChanged.connect(self.configChanged.emit)
        bbox_layout.addWidget(self.collect_bbox)
        self.bbox_widget.setLayout(bbox_layout)
        self.bbox_widget.setVisible(False)
        
        # Transform group
        self.transform = TransformWidget()
        
        # Attributes group with fixed layout
        self.attributes = QGroupBox("Attributes")
        self.attributes_layout = QVBoxLayout()
        self.attributes_layout.setSpacing(5)  # Set fixed spacing between attributes
        self.attributes_layout.setContentsMargins(10, 10, 10, 10)  # Set fixed margins
        self.attributes.setLayout(self.attributes_layout)
        
        # Add a widget to contain the attributes with scroll capability
        attributes_scroll = QScrollArea()
        attributes_scroll.setWidget(self.attributes)
        attributes_scroll.setWidgetResizable(True)
        attributes_scroll.setFixedHeight(150)  # Fixed height for attributes section
        
        # Delete button
        delete_btn = QPushButton("Delete Sensor")
        delete_btn.clicked.connect(lambda: self.deleteRequested.emit(self))
        
        # Name field and type selector containers with fixed heights
        for layout in [name_layout, type_layout]:
            container = QWidget()
            container.setLayout(layout)
            container.setFixedHeight(40)  # Set fixed height for consistent spacing
            self.layout.addWidget(container)
        
        # Bbox widget with fixed height when visible
        self.bbox_widget.setFixedHeight(40)  # Set fixed height
        self.layout.addWidget(self.bbox_widget)
        
        self.layout.addWidget(attributes_scroll)  # Attributes first
        self.layout.addWidget(self.transform)   # Transform second
        self.layout.addWidget(delete_btn)
        
        self.setLayout(self.layout)
        
        # Connect transform signal
        self.transform.configChanged.connect(self.configChanged.emit)
        
        # Initialize attributes for default sensor type
        self._on_type_changed(self.type.currentText())
    
    def _on_type_changed(self, sensor_type):
        """Handle sensor type changes"""
        # Update bbox widget visibility
        self.bbox_widget.setVisible(sensor_type == "Camera")
        # Update attributes
        self._update_attributes()
        self.configChanged.emit()
    
    def _add_camera_attributes(self):
        self.attributes_dict = {}
        attributes = [
            ("image_size_x", "Image Width", 1, 4096, 1280),
            ("image_size_y", "Image Height", 1, 4096, 720),
            ("fov", "FOV", 1, 180, 90)
        ]
        for attr_name, label, min_val, max_val, default in attributes:
            spinbox = self._add_spinbox(label, min_val, max_val, default)
            self.attributes_dict[attr_name] = spinbox
    
    def _add_radar_attributes(self):
        self.attributes_dict = {}
        attributes = [
            ("horizontal_fov", "Horizontal FOV", 1, 180, 90),
            ("vertical_fov", "Vertical FOV", 1, 90, 10),
            ("points_per_second", "Points/Second", 100, 10000, 5000),
            ("range", "Range", 1, 1000, 250)
        ]
        for attr_name, label, min_val, max_val, default in attributes:
            spinbox = self._add_spinbox(label, min_val, max_val, default)
            self.attributes_dict[attr_name] = spinbox
    
    def _add_lidar_attributes(self):
        self.attributes_dict = {}
        attributes = [
            ("channels", "Channels", 1, 128, 64),
            ("range", "Range", 1, 1000, 100),
            ("points_per_second", "Points/Second", 1000, 500000, 250000),
            ("rotation_frequency", "Rotation Frequency", 1, 100, 20)
        ]
        for attr_name, label, min_val, max_val, default in attributes:
            spinbox = self._add_spinbox(label, min_val, max_val, default)
            self.attributes_dict[attr_name] = spinbox
    
    def _add_semantic_lidar_attributes(self):
        self.attributes_dict = {}
        attributes = [
            ("channels", "Channels", 1, 128, 64),
            ("range", "Range", 1, 1000, 100),
            ("points_per_second", "Points/Second", 1000, 500000, 250000),
            ("rotation_frequency", "Rotation Frequency", 1, 100, 20),
            ("upper_fov", "Upper FOV", -90, 90, 10),
            ("lower_fov", "Lower FOV", -90, 90, -30),
            ("horizontal_fov", "Horizontal FOV", 1, 360, 360)
        ]
        for attr_name, label, min_val, max_val, default in attributes:
            spinbox = self._add_spinbox(label, min_val, max_val, default)
            self.attributes_dict[attr_name] = spinbox
    
    def _add_gnss_attributes(self):
        self.attributes_dict = {}
        noise_params = [
            ("noise_alt_bias", "Altitude Bias", 0, 1, 0.0),
            ("noise_alt_stddev", "Altitude StdDev", 0, 1, 0.1),
            ("noise_lat_bias", "Latitude Bias", 0, 1, 0.0),
            ("noise_lat_stddev", "Latitude StdDev", 0, 1, 0.1),
            ("noise_lon_bias", "Longitude Bias", 0, 1, 0.0),
            ("noise_lon_stddev", "Longitude StdDev", 0, 1, 0.1)
        ]
        for attr_name, label, min_val, max_val, default in noise_params:
            spinbox = self._add_double_spinbox(label, min_val, max_val, default)
            self.attributes_dict[attr_name] = spinbox
    
    def _add_imu_attributes(self):
        self.attributes_dict = {}
        noise_params = [
            ("noise_accel_stddev_x", "Accel StdDev X", 0, 1, 0.1),
            ("noise_accel_stddev_y", "Accel StdDev Y", 0, 1, 0.1),
            ("noise_accel_stddev_z", "Accel StdDev Z", 0, 1, 0.1),
            ("noise_gyro_stddev_x", "Gyro StdDev X", 0, 1, 0.1),
            ("noise_gyro_stddev_y", "Gyro StdDev Y", 0, 1, 0.1),
            ("noise_gyro_stddev_z", "Gyro StdDev Z", 0, 1, 0.1),
            ("noise_gyro_bias_x", "Gyro Bias X", 0, 1, 0.0),
            ("noise_gyro_bias_y", "Gyro Bias Y", 0, 1, 0.0),
            ("noise_gyro_bias_z", "Gyro Bias Z", 0, 1, 0.0)
        ]
        for attr_name, label, min_val, max_val, default in noise_params:
            spinbox = self._add_double_spinbox(label, min_val, max_val, default)
            self.attributes_dict[attr_name] = spinbox
    
    def get_config(self):
        """Return the sensor configuration as a dictionary with specific order"""
        sensor_type = self.type.currentText()
        
        # Create ordered dictionary
        config = {
            "name": self.name.text(),
            "type": sensor_type.lower().replace(" ", "_"),
            "blueprint": self.SENSOR_TYPES[sensor_type],
            "attributes": {name: str(spinbox.value()) 
                         for name, spinbox in self.attributes_dict.items()},
            "transform": self.transform.get_config()
        }
        
        # Add collect_bbox for cameras only if enabled
        if sensor_type == "Camera":
            config["collect_bbox"] = self.collect_bbox.isChecked()
        
        return config
    
    def _update_attributes(self):
        """Update attributes based on sensor type"""
        # Store the current sensor type
        sensor_type = self.type.currentText()
        
        # Clear existing attributes
        while self.attributes_layout.count():
            item = self.attributes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Reset attributes dictionary
        self.attributes_dict = {}
        
        # Add new attributes based on sensor type
        if sensor_type == "Camera":
            self._add_camera_attributes()
        elif sensor_type == "Radar":
            self._add_radar_attributes()
        elif sensor_type == "LIDAR":
            self._add_lidar_attributes()
        elif sensor_type == "Semantic LIDAR":
            self._add_semantic_lidar_attributes()
        elif sensor_type == "GNSS":
            self._add_gnss_attributes()
        elif sensor_type == "IMU":
            self._add_imu_attributes()
    
    def _add_spinbox(self, label, min_val, max_val, default):
        """Add a new integer spinbox with label"""
        container = QWidget()
        container.setFixedHeight(30)  # Fixed height for each attribute row
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        label_widget = QLabel(label)
        label_widget.setFixedWidth(120)  # Fixed width for labels
        layout.addWidget(label_widget)
        
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default)
        # Disable wheel and set focus policy
        spinbox.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        spinbox.wheelEvent = lambda event: None
        # Force dot as decimal separator
        locale = QLocale('C')  # C locale uses dot as decimal separator
        locale.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)
        spinbox.setLocale(locale)
        layout.addWidget(spinbox)
        
        container.setLayout(layout)
        self.attributes_layout.addWidget(container)
        spinbox.valueChanged.connect(self.configChanged.emit)
        return spinbox

    def _add_double_spinbox(self, label, min_val, max_val, default):
        """Add a new float spinbox with label"""
        container = QWidget()
        container.setFixedHeight(30)  # Fixed height for each attribute row
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        label_widget = QLabel(label)
        label_widget.setFixedWidth(120)  # Fixed width for labels
        layout.addWidget(label_widget)
        
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default)
        spinbox.setDecimals(3)
        # Disable wheel and set focus policy
        spinbox.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        spinbox.wheelEvent = lambda event: None
        # Force dot as decimal separator
        locale = QLocale('C')  # C locale uses dot as decimal separator
        locale.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)
        spinbox.setLocale(locale)
        layout.addWidget(spinbox)
        
        container.setLayout(layout)
        self.attributes_layout.addWidget(container)
        spinbox.valueChanged.connect(self.configChanged.emit)
        return spinbox

class LocationWidget(QGroupBox):
    """Widget for location configuration"""
    valueChanged = pyqtSignal()  # Add signal
    
    def __init__(self):
        super().__init__("Location")
        layout = QHBoxLayout()
        
        self.x = QDoubleSpinBox()
        self.y = QDoubleSpinBox()
        self.z = QDoubleSpinBox()
        
        for spinbox in [self.x, self.y, self.z]:
            spinbox.setRange(-1000, 1000)
            spinbox.setValue(0)
        
        layout.addWidget(QLabel("X:"))
        layout.addWidget(self.x)
        layout.addWidget(QLabel("Y:"))
        layout.addWidget(self.y)
        layout.addWidget(QLabel("Z:"))
        layout.addWidget(self.z)
        
        self.setLayout(layout)
        
        # Connect spinbox signals
        for spinbox in [self.x, self.y, self.z]:
            spinbox.valueChanged.connect(self.valueChanged.emit)
    
    def get_config(self):
        return {
            "x": self.x.value(),
            "y": self.y.value(),
            "z": self.z.value()
        }

class RotationWidget(QGroupBox):
    """Widget for rotation configuration"""
    valueChanged = pyqtSignal()  # Add signal
    
    def __init__(self):
        super().__init__("Rotation")
        layout = QHBoxLayout()
        
        self.pitch = QDoubleSpinBox()
        self.yaw = QDoubleSpinBox()
        self.roll = QDoubleSpinBox()
        
        for spinbox in [self.pitch, self.yaw, self.roll]:
            spinbox.setRange(-180, 180)
            spinbox.setValue(0)
        
        layout.addWidget(QLabel("Pitch:"))
        layout.addWidget(self.pitch)
        layout.addWidget(QLabel("Yaw:"))
        layout.addWidget(self.yaw)
        layout.addWidget(QLabel("Roll:"))
        layout.addWidget(self.roll)
        
        self.setLayout(layout)
        
        # Connect spinbox signals
        for spinbox in [self.pitch, self.yaw, self.roll]:
            spinbox.valueChanged.connect(self.valueChanged.emit)
    
    def get_config(self):
        return {
            "pitch": self.pitch.value(),
            "yaw": self.yaw.value(),
            "roll": self.roll.value()
        }

class MainWindow(QMainWindow):
    """Main window of the configuration editor"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CARLA Simulation Configuration Editor")
        self.setMinimumSize(1200, 800)  # Increased from 800x600
        
        # Create central widget and layout
        central_widget = QWidget()
        layout = QHBoxLayout()
        
        # Create tab widget for simulation and sensor settings
        tabs = QTabWidget()
        self.sim_tab = SimulationTab()
        self.sensor_tab = SensorTab()
        tabs.addTab(self.sim_tab, "Simulation")
        tabs.addTab(self.sensor_tab, "Sensors")
        
        # Create YAML preview
        preview_group = QGroupBox("YAML Preview")
        preview_layout = QVBoxLayout()
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        preview_layout.addWidget(self.preview)
        preview_group.setLayout(preview_layout)
        
        # Add save, run and visualize buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save Configuration")
        run_btn = QPushButton("Run Simulation")
        visualize_btn = QPushButton("Visualize Simulation")
        save_btn.clicked.connect(self.save_config)
        run_btn.clicked.connect(self.run_simulation)
        visualize_btn.clicked.connect(self.visualize_simulation)
        button_layout.addWidget(save_btn)
        button_layout.addWidget(run_btn)
        button_layout.addWidget(visualize_btn)
        
        # Add widgets to layout
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(tabs)
        left_layout.addLayout(button_layout)
        left_panel.setLayout(left_layout)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(preview_group)
        # Set initial proportions to 70/30
        splitter.setStretchFactor(0, 7)  # Left panel gets 70%
        splitter.setStretchFactor(1, 3)  # Preview gets 30%
        
        layout.addWidget(splitter)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        # Connect signals for live preview updates
        self.sim_tab.configChanged.connect(self.update_preview)
        self.sensor_tab.configChanged.connect(self.update_preview)
        
        # Initial preview update
        self.update_preview()
    
    def update_preview(self):
        """Update the YAML preview while maintaining scroll position"""
        try:
            # Store current scroll position
            scrollbar = self.preview.verticalScrollBar()
            current_pos = scrollbar.value()
            
            # Update YAML content
            sim_config = self.sim_tab.get_config()
            sensors_config = self.sensor_tab.get_config()
            config = {
                "simulation": sim_config["simulation"],
                "sensors": sensors_config
            }
            yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
            
            # Update text and restore scroll position
            self.preview.setPlainText(yaml_str)
            scrollbar.setValue(current_pos)
            
        except Exception as e:
            self.preview.setPlainText(f"Error generating YAML: {str(e)}")
    
    def save_config(self):
        """Save the configuration to a YAML file"""
        try:
            # Set default filename to config.yml
            default_filename = "config.yml"
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Configuration",
                default_filename,
                "YAML Files (*.yml *.yaml)"
            )
            
            # Only save if user selected a file path
            if file_path:
                config = {
                    "simulation": self.sim_tab.get_config()["simulation"],
                    "sensors": self.sensor_tab.get_config()
                }
                with open(file_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                QMessageBox.information(self, "Success", 
                                    f"Configuration saved to {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {str(e)}")
    
    def run_simulation(self):
        """Save the configuration and run the simulation"""
        try:
            # Check if config.yml exists, if not prompt user to save first
            if not os.path.exists('config.yml'):
                QMessageBox.warning(
                    self, 
                    "Configuration Missing",
                    "Please save your configuration first using the Save Configuration button."
                )
                return
            
            # Get current script directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Get Python executable from current environment
            python_exe = sys.executable
            
            # Create process with proper working directory
            process = subprocess.Popen(
                [python_exe, "multi_sensor_collection.py"],
                cwd=current_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Show a more informative dialog
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Simulation is running...\n\n" + 
                       "Check the console for output.\n" +
                       "The simulation may take several minutes to complete.")
            msg.setWindowTitle("Simulation Status")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            
            # Get results (non-blocking)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Simulation failed:\n{stderr}")
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to run simulation: {str(e)}\n\n" +
                "Make sure CARLA simulator is running!"
            )
    
    def visualize_simulation(self):
        """Show scene selection dialog and run visualization"""
        try:
            # Get output directory from config
            base_save_path = self.sim_tab.get_config()["simulation"]["base_save_path"]
            
            # List available scenes
            scene_dir = Path(os.path.abspath(base_save_path))  # Fixed path resolution
            if not scene_dir.exists():
                raise RuntimeError(f"Output directory not found: {base_save_path}")
            
            scenes = sorted([d.name for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("scene_")])
            
            if not scenes:
                raise RuntimeError("No scenes found in output directory")
            
            # Create scene selection dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Select Scene to Visualize")
            dialog.setModal(True)  # Make dialog modal
            layout = QVBoxLayout()
            
            # Add scene selector
            label = QLabel("Select scene to visualize:")
            combo = QComboBox()
            combo.addItems(scenes)
            
            # Add buttons
            buttons = QHBoxLayout()
            ok_btn = QPushButton("Visualize")
            cancel_btn = QPushButton("Cancel")
            ok_btn.clicked.connect(dialog.accept)
            cancel_btn.clicked.connect(dialog.reject)
            buttons.addWidget(ok_btn)
            buttons.addWidget(cancel_btn)
            
            # Arrange dialog layout
            layout.addWidget(label)
            layout.addWidget(combo)
            layout.addLayout(buttons)
            dialog.setLayout(layout)
            
            # Show dialog and handle result
            if dialog.exec() == 1:  # Use 1 for Accepted instead of enum
                selected_scene = combo.currentText()
                
                # Get Python executable and current directory
                python_exe = sys.executable
                current_dir = os.path.dirname(os.path.abspath(__file__))
                
                # Run visualization script
                process = subprocess.Popen(
                    [python_exe, "multi_sensor_replay.py", selected_scene],
                    cwd=current_dir
                )
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to start visualization: {str(e)}"
            )

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
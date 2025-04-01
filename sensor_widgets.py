from PyQt6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, 
                            QDoubleSpinBox, QSpinBox, QWidget, QComboBox,
                            QPushButton, QCheckBox, QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal, QLocale

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
        "Semantic Camera": "sensor.camera.semantic_segmentation",
        "Instance Camera": "sensor.camera.instance_segmentation",
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
        
        # Name field and type selector containers with fixed heights
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name = QLineEdit("new_sensor")
        self.name.textChanged.connect(self.configChanged.emit)
        name_layout.addWidget(self.name)
        
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type = QComboBox()
        self.type.addItems(self.SENSOR_TYPES.keys())
        self.type.currentTextChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.type)
        
        # Create containers with fixed heights
        for layout_item in [name_layout, type_layout]:
            container = QWidget()
            container.setLayout(layout_item)
            container.setFixedHeight(40)
            self.layout.addWidget(container)

        # Attributes group
        self.attributes = QGroupBox("Attributes")
        self.attributes_layout = QVBoxLayout()
        self.attributes_layout.setSpacing(5)
        self.attributes_layout.setContentsMargins(10, 10, 10, 10)
        self.attributes.setLayout(self.attributes_layout)
        
        # BBox widget
        self.bbox_widget = QWidget()
        bbox_layout = QHBoxLayout()
        self.collect_bbox = QCheckBox("Enable Bounding Box Collection")
        self.collect_bbox.setChecked(True)
        self.collect_bbox.stateChanged.connect(self.configChanged.emit)
        bbox_layout.addWidget(self.collect_bbox)
        self.bbox_widget.setLayout(bbox_layout)
        self.bbox_widget.setFixedHeight(40)
        self.bbox_widget.setVisible(False)

        # Transform widget
        self.transform = TransformWidget()
        
        # Delete button
        delete_btn = QPushButton("Delete Sensor")
        delete_btn.clicked.connect(lambda: self.deleteRequested.emit(self))

        # Add widgets in the desired order
        self.layout.addWidget(self.bbox_widget)
        self.layout.addWidget(self.attributes)  
        self.layout.addWidget(self.transform)
        self.layout.addWidget(delete_btn)
        
        self.setLayout(self.layout)
        
        # Connect signals and initialize
        self.transform.configChanged.connect(self.configChanged.emit)
        self._on_type_changed(self.type.currentText())
    
    def _on_type_changed(self, sensor_type):
        """Handle sensor type changes"""
        # Update bbox widget visibility - only for RGB cameras
        self.bbox_widget.setVisible(sensor_type == "Camera")
        # Update attributes
        self._update_attributes()
        self.configChanged.emit()
    
    def _add_basic_camera_attributes(self):
        """Add basic camera attributes (for all camera types)"""
        self.attributes_dict = {}
        
        # Basic attributes
        # basic_label = QLabel("Basic Settings")
        # basic_label.setStyleSheet("font-weight: bold;")
        # self.attributes_layout.addWidget(basic_label)
        
        basic_attributes = [
            ("image_size_x", "Image Width", 1, 4096, 800),
            ("image_size_y", "Image Height", 1, 4096, 600),
            ("fov", "FOV", 1, 180, 90.0)
        ]
        
        for attr_name, label, min_val, max_val, default in basic_attributes:
            spinbox = self._add_double_spinbox(label, min_val, max_val, default)
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
            ("rotation_frequency", "Rotation Frequency", 1, 100, 20.0),
            ("upper_fov", "Upper FOV", -90, 90, 10.0),
            ("lower_fov", "Lower FOV", -90, 90, -30.0),
            ("horizontal_fov", "Horizontal FOV", 1, 360, 360.0),
            ("atmosphere_attenuation_rate", "Atmosphere Attenuation", 0.0, 1.0, 0.004), 
            ("dropoff_general_rate", "General Dropoff Rate", 0.0, 1.0, 0.45),
            ("dropoff_intensity_limit", "Intensity Dropoff Limit", 0.0, 1.0, 0.8),
            ("dropoff_zero_intensity", "Zero Intensity Dropoff", 0.0, 1.0, 0.4),
            ("noise_stddev", "Noise StdDev", 0.0, 1.0, 0.0)
        ]
        for attr_name, label, min_val, max_val, default in attributes:
            spinbox = self._add_double_spinbox(label, min_val, max_val, default)
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
            ("noise_gyro_stddev_y", "Accel StdDev Y", 0, 1, 0.1),
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
            "transform": {
                "location": self.transform.location.get_config(),
                "rotation": {
                    "pitch": self.transform.rotation.pitch.value(),
                    "yaw": self.transform.rotation.yaw.value(),
                    "roll": self.transform.rotation.roll.value()
                }
            }
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
        if sensor_type in ["Camera", "Semantic Camera", "Instance Camera"]:  
            self._add_basic_camera_attributes()
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
            
        # Update the widget's size after changing attributes
        self.attributes.adjustSize()
        self.adjustSize()
    
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
        
        # Extend range to allow full rotation
        self.pitch.setRange(-360, 360)
        self.yaw.setRange(-360, 360)  # Allow full rotation range
        self.roll.setRange(-360, 360)
        
        # Set default values
        for spinbox in [self.pitch, self.yaw, self.roll]:
            spinbox.setValue(0)
            spinbox.setDecimals(1)  # Show one decimal place for precision
            # Disable wheel and set focus policy
            spinbox.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            spinbox.wheelEvent = lambda event: None
        
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

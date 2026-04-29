from PyQt6.QtWidgets import (QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QSpinBox, 
                            QCheckBox, QPushButton, QGroupBox,
                            QFileDialog)  
from PyQt6.QtCore import Qt, pyqtSignal

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
        
        seconds_container = self._create_spinbox("Seconds per Scene:", 1, 1000, 20)
        self.seconds_per_scene = seconds_container.findChild(QSpinBox)
        
        # Simulation frequency (Hz)
        frequency_container = self._create_spinbox("Frequency (Hz):", 1, 200, 20)
        self.frequency_hz = frequency_container.findChild(QSpinBox)
        
        self.base_save_path = self._create_path_selector("Base Save Path:", "./data/_out")
        
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
        for widget in [num_scenes_container, seconds_container, frequency_container, 
                      self.base_save_path, traffic_group]:
            sim_layout.addWidget(widget)
        
        sim_group.setLayout(sim_layout)
        self.layout.addWidget(sim_group)
        self.layout.addStretch()
        self.setLayout(self.layout)
        
        # Connect signals
        for spinbox in [self.num_scenes, self.seconds_per_scene, self.frequency_hz, 
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
    
    def load_config(self, cfg):
        """Populate widgets from a config dict (simulation + traffic keys)."""
        sim = cfg.get("simulation", cfg)  # accept top-level or nested
        traffic = sim.get("traffic", {})

        # Block signals while loading to avoid spurious configChanged emissions
        widgets = [self.num_scenes, self.seconds_per_scene, self.frequency_hz,
                   self.num_vehicles, self.num_pedestrians,
                   self.safe_spawn, self.car_lights_on, self.path_edit]
        for w in widgets:
            w.blockSignals(True)

        self.num_scenes.setValue(int(sim.get("num_scenes", self.num_scenes.value())))
        self.seconds_per_scene.setValue(int(sim.get("seconds_per_scene", self.seconds_per_scene.value())))
        self.frequency_hz.setValue(int(sim.get("frequency_hz", self.frequency_hz.value())))
        self.path_edit.setText(str(sim.get("base_save_path", self.path_edit.text())))
        self.num_vehicles.setValue(int(traffic.get("num_vehicles", self.num_vehicles.value())))
        self.num_pedestrians.setValue(int(traffic.get("num_pedestrians", self.num_pedestrians.value())))
        self.safe_spawn.setChecked(bool(traffic.get("safe_spawn", self.safe_spawn.isChecked())))
        self.car_lights_on.setChecked(bool(traffic.get("car_lights_on", self.car_lights_on.isChecked())))

        for w in widgets:
            w.blockSignals(False)

    def get_config(self):
        """Return the current configuration as a dictionary"""
        return {
            "simulation": {
                "num_scenes": self.num_scenes.value(),
                "seconds_per_scene": self.seconds_per_scene.value(),
                "frequency_hz": self.frequency_hz.value(),
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

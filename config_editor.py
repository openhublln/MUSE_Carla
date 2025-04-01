import sys
import yaml
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTabWidget, QLabel, QPushButton,
                            QGroupBox, QTextEdit, QSplitter, QDialog, QMessageBox,
                            QFileDialog, QComboBox)
from PyQt6.QtCore import Qt
import os
import subprocess
from pathlib import Path
from simulation_tab import SimulationTab
from sensor_tab import SensorTab

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
        launch_btn = QPushButton("Launch CARLA")
        save_btn = QPushButton("Save Configuration")
        run_btn = QPushButton("Run Simulation")
        visualize_btn = QPushButton("Visualize Simulation")
        
        launch_btn.clicked.connect(self.launch_carla)
        save_btn.clicked.connect(self.save_config)
        run_btn.clicked.connect(self.run_simulation)
        visualize_btn.clicked.connect(self.visualize_simulation)
        
        button_layout.addWidget(launch_btn)
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
            
            if process.returncode == 0:
                # Show success message
                QMessageBox.information(
                    self,
                    "Simulation Complete",
                    "The simulation has finished successfully!\n\n" +
                    "Data has been saved to the output directory."
                )
            else:
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
            
            # Create scene selection dialog with annotation type selection
            dialog = QDialog(self)
            dialog.setWindowTitle("Select Scene to Visualize")
            dialog.setModal(True)  # Make dialog modal
            layout = QVBoxLayout()
            
            # Add scene selector
            label = QLabel("Select scene to visualize:")
            combo = QComboBox()
            combo.addItems(scenes)
            layout.addWidget(label)
            layout.addWidget(combo)
            
            # Add annotation type selector
            ann_layout = QHBoxLayout()
            ann_label = QLabel("Select Bounding Box Type:")
            ann_combo = QComboBox()
            ann_combo.addItem("2D", "2d")
            ann_combo.addItem("3D", "3d")
            ann_layout.addWidget(ann_label)
            ann_layout.addWidget(ann_combo)
            layout.addLayout(ann_layout)
            
            # Add buttons
            buttons = QHBoxLayout()
            ok_btn = QPushButton("Visualize")
            cancel_btn = QPushButton("Cancel")
            ok_btn.clicked.connect(dialog.accept)
            cancel_btn.clicked.connect(dialog.reject)
            buttons.addWidget(ok_btn)
            buttons.addWidget(cancel_btn)
            layout.addLayout(buttons)
            dialog.setLayout(layout)
            
            # Show dialog and handle result
            if dialog.exec() == QDialog.DialogCode.Accepted:
                selected_scene = combo.currentText()
                annotation_type = ann_combo.currentData()
                
                # Get Python executable and current directory
                python_exe = sys.executable
                current_dir = os.path.dirname(os.path.abspath(__file__))
                
                # Run visualization script with scene and annotation type arguments
                process = subprocess.Popen(
                    [python_exe, "multi_sensor_replay.py", selected_scene, annotation_type],
                    cwd=current_dir
                )
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to start visualization: {str(e)}"
            )
    
    def launch_carla(self):
        """Launch CARLA server using relative path"""
        try:
            # Get the path to CARLA by going up from current directory
            current_dir = Path(os.path.abspath(__file__))  # Get path to current script
            carla_root = current_dir.parents[2]  # Go up 3 levels to CARLA root
            carla_exe = carla_root / "CarlaUnreal.exe"
            
            if not carla_exe.exists():
                raise FileNotFoundError(f"CARLA executable not found at: {carla_exe}")
            
            # Launch CARLA in a new process
            subprocess.Popen(
                str(carla_exe),
                cwd=str(carla_root),
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            
            # Show a message to wait for CARLA to start
            QMessageBox.information(
                self,
                "CARLA Starting",
                "CARLA is starting...\n\n" +
                "Please wait for the simulator to fully load before running the simulation."
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to launch CARLA: {str(e)}\n\n" +
                "Please make sure CARLA is correctly installed."
            )

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
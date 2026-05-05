import sys
import yaml
import tempfile
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTabWidget, QLabel, QPushButton,
                            QGroupBox, QTextEdit, QSplitter, QDialog, QMessageBox,
                            QFileDialog, QComboBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt
import os
import subprocess
from pathlib import Path
from gui.simulation_tab import SimulationTab
from gui.sensor_tab import SensorTab


def _proc_alive(pid):
    """Return True if the process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours to signal

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
        self._preview_group = QGroupBox("YAML Preview")
        preview_layout = QVBoxLayout()
        preview_layout.setSpacing(4)

        # Unsaved indicator — always present (fixed height), so layout never reflows.
        # Text and colour toggle; invisible state shows blank.
        self._unsaved_label = QLabel("")
        self._unsaved_label.setFixedHeight(18)
        self._unsaved_label.setStyleSheet("color: transparent;")  # invisible by default
        preview_layout.addWidget(self._unsaved_label)

        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        preview_layout.addWidget(self.preview)
        self._preview_group.setLayout(preview_layout)
        preview_group = self._preview_group  # alias for splitter below
        
        # Add save, run and visualize buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save Configuration")
        run_btn = QPushButton("Run Simulation")
        visualize_btn = QPushButton("Visualize Simulation")
        convert_nuscene_btn = QPushButton("Convert to NuScenes")

        save_btn.clicked.connect(self.save_config)
        run_btn.clicked.connect(self.run_simulation)
        visualize_btn.clicked.connect(self.visualize_simulation)
        convert_nuscene_btn.clicked.connect(self.convert_to_nuscene)

        button_layout.addWidget(save_btn)
        button_layout.addWidget(run_btn)
        button_layout.addWidget(visualize_btn)
        button_layout.addWidget(convert_nuscene_btn)
        
        # Add widgets to layout
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(tabs)
        left_layout.addLayout(button_layout)
        left_panel.setLayout(left_layout)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(preview_group)
        # Set initial proportions to 60/40
        splitter.setStretchFactor(0, 6)  # Left panel gets 60%
        splitter.setStretchFactor(1, 4)  # Preview gets 40%
        
        layout.addWidget(splitter)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        # Connect signals for live preview updates
        self.sim_tab.configChanged.connect(self.update_preview)
        self.sensor_tab.configChanged.connect(self.update_preview)

        # Load config.yml if it exists, then take initial snapshot of saved state
        self._saved_yaml = ""
        self._load_config_from_disk()
        # update_preview is called inside _load_config_from_disk (via configChanged),
        # so _saved_yaml is set there.  If the file was absent, do a plain initial update.
        if not self._saved_yaml:
            self.update_preview()
            self._saved_yaml = self.preview.toPlainText()
    
    def _current_yaml(self):
        """Return YAML string for the current widget state."""
        sim_config = self.sim_tab.get_config()
        sensors_config = self.sensor_tab.get_config()
        config = {
            "simulation": sim_config["simulation"],
            "sensors": sensors_config
        }
        return yaml.dump(config, default_flow_style=False, sort_keys=False)

    def _set_dirty(self, dirty):
        """Toggle the unsaved-changes indicator without affecting layout."""
        if dirty:
            self._unsaved_label.setText("● unsaved changes")
            self._unsaved_label.setStyleSheet("color: #c0702a; font-style: italic; font-size: 11px;")
        else:
            self._unsaved_label.setText("")
            self._unsaved_label.setStyleSheet("color: transparent;")

    def _load_config_from_disk(self):
        """Load config.yml into tabs if it exists."""
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")
        if not os.path.exists(config_path):
            return
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            if not cfg:
                return
            # Populate tabs (signals blocked internally)
            self.sim_tab.load_config(cfg)
            self.sensor_tab.load_config(cfg.get("sensors", []))
            # Snapshot the saved state
            saved = self._current_yaml()
            self._saved_yaml = saved
            # Update preview without marking as unsaved
            scrollbar = self.preview.verticalScrollBar()
            pos = scrollbar.value()
            self.preview.setPlainText(saved)
            scrollbar.setValue(pos)
            self._set_dirty(False)
        except Exception as e:
            print(f"Warning: could not load config.yml: {e}")

    def update_preview(self):
        """Update the YAML preview while maintaining scroll position"""
        try:
            scrollbar = self.preview.verticalScrollBar()
            current_pos = scrollbar.value()

            yaml_str = self._current_yaml()
            self.preview.setPlainText(yaml_str)
            scrollbar.setValue(current_pos)

            # Visual feedback: unsaved changes
            is_dirty = (yaml_str != self._saved_yaml)
            self._set_dirty(is_dirty)
        except Exception as e:
            self.preview.setPlainText(f"Error generating YAML: {str(e)}")

    def save_config(self):
        """Save the current configuration to config.yml."""
        try:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")
            config = {
                "simulation": self.sim_tab.get_config()["simulation"],
                "sensors": self.sensor_tab.get_config()
            }
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            # Update saved snapshot and clear indicator
            self._saved_yaml = self._current_yaml()
            self._set_dirty(False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {str(e)}")
    
    def _find_carla_executable(self):
        """Return path to CarlaUnreal.sh/.exe, or raise FileNotFoundError."""
        carla_root = Path(os.path.abspath(__file__)).parents[3]
        if sys.platform == "win32":
            exe = carla_root / "CarlaUnreal.exe"
            if not exe.exists():
                raise FileNotFoundError(f"CARLA executable not found: {exe}")
            return exe, carla_root
        else:
            for name in ("CarlaUnreal.sh", "CarlaUnreal"):
                exe = carla_root / name
                if exe.exists():
                    return exe, carla_root
            raise FileNotFoundError(
                f"CARLA executable not found in {carla_root} "
                "(tried CarlaUnreal.sh and CarlaUnreal)"
            )

    def _wait_for_carla(self, carla_proc, host="localhost", port=2000, timeout=120):
        """Block until CARLA accepts TCP connections on port 2000, or timeout, or CARLA crashes.

        Returns (True, None) if ready.
        Returns (False, "timeout") if the timeout expired without a connection.
        Returns (False, "crashed") if carla_proc exited before the port opened.
        """
        import socket, time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Check if CARLA already died
            if carla_proc.poll() is not None:
                return False, "crashed"
            try:
                with socket.create_connection((host, port), timeout=2):
                    # Port is open — do one final liveness check before returning
                    if carla_proc.poll() is not None:
                        return False, "crashed"
                    return True, None
            except OSError:
                time.sleep(2)
        return False, "timeout"

    def _kill_existing_carla(self, port=2000, wait=20):
        """Kill any CarlaUnreal process already holding port 2000 before launching a fresh one.
        Waits until the port is actually free (not just until the process exits).
        """
        import time, signal as _signal

        def _pids_on_port():
            try:
                result = subprocess.run(
                    ["lsof", "-t", f"-i:{port}"],
                    capture_output=True, text=True
                )
                return [int(p) for p in result.stdout.split() if p.strip().isdigit()]
            except Exception:
                return []

        def _port_free():
            import socket
            try:
                with socket.create_connection(("localhost", port), timeout=1):
                    return False
            except OSError:
                return True

        pids = _pids_on_port()
        if not pids and _port_free():
            return  # nothing to do

        if pids:
            print(f"[muse] Killing stale CARLA PID(s) on port {port}: {pids}")
            for pid in pids:
                try:
                    os.kill(pid, _signal.SIGTERM)
                except ProcessLookupError:
                    pass

        # Wait for port to be free (not just process exit — kernel needs to release it)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            time.sleep(1)
            if _port_free():
                return

        # Still not free — SIGKILL remaining holders
        pids = _pids_on_port()
        for pid in pids:
            try:
                os.kill(pid, _signal.SIGKILL)
            except ProcessLookupError:
                pass

        # Final wait
        deadline2 = time.monotonic() + 5
        while time.monotonic() < deadline2:
            time.sleep(1)
            if _port_free():
                return

        print(f"[muse] WARNING: port {port} may still be in use after kill attempts")

    def _wait_for_carla_rpc(self, carla_proc, host="localhost", port=2000, timeout=60):
        """After the TCP port is open, probe the CARLA RPC until get_world() succeeds.

        Returns (True, None) when the world is fully loaded.
        Returns (False, "crashed") if carla_proc exits while waiting.
        Returns (False, "timeout") if timeout expires.
        """
        import time
        try:
            import carla as _carla
        except ImportError:
            # carla not importable in GUI process — fall back to a fixed sleep
            time.sleep(15)
            if carla_proc.poll() is not None:
                return False, "crashed"
            return True, None

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if carla_proc.poll() is not None:
                return False, "crashed"
            try:
                c = _carla.Client(host, port)
                c.set_timeout(3.0)
                c.get_world()
                return True, None
            except Exception:
                time.sleep(2)
        return False, "timeout"

    def run_simulation(self):
        """Launch CARLA headless (-RenderOffScreen), wait until ready, run collection, then shut CARLA down."""
        import time, signal

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.yml")

        if not os.path.exists(config_path):
            QMessageBox.warning(
                self,
                "Configuration Missing",
                "Please save your configuration first using the Save Configuration button."
            )
            return

        carla_proc = None
        try:
            # --- 1. Locate and launch CARLA ---
            carla_exe, carla_root = self._find_carla_executable()
            carla_log = carla_root / "carla_launch.log"

            # Kill any leftover CARLA process before launching a new one
            self._kill_existing_carla()

            with open(carla_log, "w") as lf:
                carla_proc = subprocess.Popen(
                    [str(carla_exe), "-RenderOffScreen"],
                    cwd=str(carla_root),
                    stdout=lf,
                    stderr=lf,
                    start_new_session=True
                )

            # --- 2. Wait for CARLA to be ready ---
            QMessageBox.information(
                self,
                "CARLA Starting",
                "CARLA is starting in headless mode (-RenderOffScreen).\n\n"
                "Click OK — the GUI will wait up to 2 minutes for CARLA to be ready,\n"
                "then start data collection automatically."
            )

            ready, reason = self._wait_for_carla(carla_proc, timeout=120)
            if not ready:
                if reason == "crashed":
                    raise RuntimeError(
                        "CARLA crashed during startup.\n"
                        f"Check the log for details:\n  {carla_log}"
                    )
                else:
                    raise RuntimeError(
                        "CARLA did not become ready within 2 minutes.\n"
                        f"Check the log for details:\n  {carla_log}"
                    )

            # TCP port is open but the CARLA RPC world may not be ready yet.
            # Actively probe get_world() until it succeeds (or timeout / crash).
            rpc_ready, rpc_reason = self._wait_for_carla_rpc(carla_proc, timeout=60)
            if not rpc_ready:
                if rpc_reason == "crashed":
                    raise RuntimeError(
                        "CARLA crashed before the RPC world became ready.\n"
                        f"Check the log for details:\n  {carla_log}"
                    )
                else:
                    raise RuntimeError(
                        "CARLA RPC world did not become ready within 60 seconds after port opened.\n"
                        f"Check the log for details:\n  {carla_log}"
                    )

            # --- 3. Run data collection ---
            python_exe = sys.executable
            collection_proc = subprocess.Popen(
                [python_exe, "collection/multi_sensor_collection.py"],
                cwd=current_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            stdout, stderr = collection_proc.communicate()

            if collection_proc.returncode != 0:
                # Include stdout too — collection script prints errors there
                details = (stderr.strip() or stdout.strip() or "(no output captured)")
                raise RuntimeError(f"Simulation failed:\n{details}")

            QMessageBox.information(
                self,
                "Simulation Complete",
                "The simulation has finished successfully!\n\n"
                "Data has been saved to the output directory."
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to run simulation: {str(e)}"
            )
        finally:
            # --- 4. Always shut CARLA down ---
            if carla_proc is not None and carla_proc.poll() is None:
                try:
                    os.killpg(os.getpgid(carla_proc.pid), signal.SIGTERM)
                except Exception:
                    carla_proc.terminate()
                try:
                    carla_proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(carla_proc.pid), signal.SIGKILL)
                    except Exception:
                        carla_proc.kill()
                    try:
                        carla_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            # Ensure port 2000 is free before returning (next run's _kill_existing_carla
            # is a safety net, but better to be clean here too)
            import time as _time, socket as _socket
            for _ in range(10):
                try:
                    with _socket.create_connection(("localhost", 2000), timeout=1):
                        _time.sleep(2)
                except OSError:
                    break
    
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
            
            # Check whether any camera has collect_bbox: true in the current config
            sensor_list = self.sensor_tab.get_config()
            has_2d_bbox = any(
                s.get("type") == "camera" and s.get("collect_bbox", False)
                for s in sensor_list
            )

            # Add annotation type selector — 3D is always available and default;
            # 2D is only enabled when collect_bbox is active on at least one camera
            ann_layout = QHBoxLayout()
            ann_label = QLabel("Select Bounding Box Type:")
            ann_combo = QComboBox()
            ann_combo.addItem("3D", "3d")
            if has_2d_bbox:
                ann_combo.addItem("2D", "2d")
            else:
                ann_combo.addItem("2D (not configured)", "2d")
                ann_combo.model().item(1).setEnabled(False)
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
                    [python_exe, "replay/multi_sensor_replay.py", selected_scene, annotation_type],
                    cwd=current_dir
                )
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to start visualization: {str(e)}"
            )
    
    def convert_to_nuscene(self):
        """Show conversion options dialog, then run the CARLA to NuScenes conversion."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            converter_config_path = os.path.join(current_dir, "converter_config.yml")

            if not os.path.exists(converter_config_path):
                QMessageBox.warning(
                    self,
                    "Converter Configuration Missing",
                    "converter_config.yml not found in the script directory.\n"
                    "Please ensure the converter configuration file is present."
                )
                return

            # Load base config to read defaults
            with open(converter_config_path, 'r') as f:
                base_config = yaml.safe_load(f)

            # Input dir comes from config.yml (the saved collection config)
            sim_config_path = os.path.join(current_dir, "config.yml")
            if os.path.exists(sim_config_path):
                with open(sim_config_path, 'r') as f:
                    sim_config = yaml.safe_load(f)
                input_base = sim_config.get("simulation", {}).get("base_save_path", "data/_out")
            else:
                input_base = base_config.get('input', {}).get('base_dir', 'data/_out')
            default_rate = float(base_config.get('output', {}).get('keyframe_rate', 2.0))

            # --- Conversion options dialog ---
            dialog = QDialog(self)
            dialog.setWindowTitle("NuScenes Conversion Options")
            dialog.setModal(True)
            dlg_layout = QVBoxLayout()

            # Input dir (informational, read-only)
            input_row = QHBoxLayout()
            input_row.addWidget(QLabel("Input data:"))
            input_label = QLabel(input_base)
            input_label.setStyleSheet("color: grey;")
            input_row.addWidget(input_label)
            input_row.addStretch()
            dlg_layout.addLayout(input_row)

            # Keyframe rate spinner
            rate_row = QHBoxLayout()
            rate_row.addWidget(QLabel("Keyframe rate (Hz):"))
            rate_spin = QDoubleSpinBox()
            rate_spin.setRange(0.1, 100.0)
            rate_spin.setSingleStep(1.0)
            rate_spin.setDecimals(1)
            rate_spin.setValue(default_rate)
            rate_row.addWidget(rate_spin)
            rate_row.addStretch()
            dlg_layout.addLayout(rate_row)

            # Output folder preview (auto-updates)
            out_row = QHBoxLayout()
            out_row.addWidget(QLabel("Output folder:"))

            def _fmt_rate(hz):
                """Format Hz value as a clean string for folder naming."""
                return f"{hz:g}Hz"

            out_preview = QLabel()
            out_preview.setStyleSheet("color: grey;")

            def _update_preview():
                rate = rate_spin.value()
                folder = f"data/nuscenes_{_fmt_rate(rate)}"
                out_preview.setText(folder)

            rate_spin.valueChanged.connect(_update_preview)
            _update_preview()
            out_row.addWidget(out_preview)
            out_row.addStretch()
            dlg_layout.addLayout(out_row)

            # Buttons
            btn_row = QHBoxLayout()
            ok_btn = QPushButton("Convert")
            cancel_btn = QPushButton("Cancel")
            ok_btn.clicked.connect(dialog.accept)
            cancel_btn.clicked.connect(dialog.reject)
            btn_row.addWidget(ok_btn)
            btn_row.addWidget(cancel_btn)
            dlg_layout.addLayout(btn_row)

            dialog.setLayout(dlg_layout)

            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            # Build runtime config: patch rate, input and output dirs, write to a temp file
            keyframe_rate = rate_spin.value()
            output_dir = f"data/nuscenes_{_fmt_rate(keyframe_rate)}"

            runtime_config = base_config.copy()
            runtime_config.setdefault('input', {})
            runtime_config['input']['base_dir'] = input_base
            runtime_config.setdefault('output', {})
            runtime_config['output']['keyframe_rate'] = keyframe_rate
            runtime_config['output']['base_dir'] = output_dir

            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.yml', delete=False, dir=current_dir,
                prefix='.converter_runtime_'
            ) as tmp:
                yaml.dump(runtime_config, tmp, default_flow_style=False, sort_keys=False)
                tmp_config_path = tmp.name

            python_exe = sys.executable

            QMessageBox.information(
                self,
                "Conversion Starting",
                f"Converting at {keyframe_rate:g} Hz → {output_dir}\n\n"
                "This may take some time. A notification will appear when done."
            )

            process = subprocess.Popen(
                [python_exe, "conversion/carla_to_nuscene_converter.py", tmp_config_path],
                cwd=current_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace"
            )

            stdout, stderr = process.communicate()

            # Clean up temp config
            try:
                os.unlink(tmp_config_path)
            except OSError:
                pass

            if process.returncode == 0:
                QMessageBox.information(
                    self,
                    "Conversion Complete",
                    f"Conversion to NuScenes format complete!\n\nOutput: {output_dir}"
                )
            else:
                QMessageBox.critical(
                    self,
                    "Conversion Failed",
                    f"Conversion failed.\n\nError:\n{stderr}"
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to run NuScenes conversion: {str(e)}"
            )

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
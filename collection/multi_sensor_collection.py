import os
import time
import yaml
import carla
import json
import subprocess
import sys
import glob
from pathlib import Path
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent.parent  # MUSE_Carla/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bounding_box_export import export_3d_bboxes, get_static_vehicle_env_objects
from traffic_setup import setup_traffic, spawn_ego_vehicle
from sensor_processing import process_sensor_config, sensor_callback, write_sensor_data, clean_scene_data
from simulation_logic import create_scene_folders
from generate_bbox_annotations import process_scene

EGO_POSE_FOLDER = "ego_pose"
LOG_INFO_FILENAME = "log_info.json"


def _next_scene_id(base_save_path: str) -> int:
    """Return the next scene ID by scanning existing scene_N folders.

    If no scene folders exist yet, returns 1.  If scene_1 and scene_2 exist,
    returns 3 — so new runs always append rather than overwrite.
    """
    base = Path(base_save_path)
    if not base.exists():
        return 1
    existing = []
    for item in base.iterdir():
        if item.is_dir() and item.name.startswith("scene_"):
            try:
                existing.append(int(item.name.split("_", 1)[1]))
            except ValueError:
                pass
    return max(existing) + 1 if existing else 1

# Worker threads for file I/O.  All data passed to workers is plain Python /
# NumPy — no CARLA C++ objects — so thread count can be higher.
IO_WORKERS = 8

# Maximum number of futures allowed in-flight before we block world.tick().
# Prevents unbounded memory growth when I/O is slower than sensor tick rate.
# At 10 Hz with ~16 sensors/tick each frame is ~2 MB; 32 pending = ~64 MB max.
MAX_PENDING_FUTURES = 32


def build_actor_static_cache(world, ego_id):
    """Fetch static actor metadata once per scene (type_id + bounding_box).

    These fields never change for a spawned actor, so we only need one RPC call
    per scene instead of one per tick.  Returns a dict keyed by actor ID.
    """
    cache = {}
    try:
        all_actors = world.get_actors()
        for actor in list(all_actors.filter('vehicle.*')) + list(all_actors.filter('walker.pedestrian.*')):
            if actor.id == ego_id:
                continue
            try:
                bb  = actor.bounding_box
                ext = bb.extent
                cache[actor.id] = {
                    'type_id': actor.type_id,
                    'bounding_box': {
                        'loc_x': bb.location.x, 'loc_y': bb.location.y, 'loc_z': bb.location.z,
                        'ext_x': ext.x, 'ext_y': ext.y, 'ext_z': ext.z,
                    },
                }
            except Exception:
                pass
    except Exception as e:
        print(f"Warning: build_actor_static_cache failed: {e}")
    return cache


def build_actor_snapshot(static_cache, world_snapshot):
    """Build per-tick actor snapshot from world_snapshot only — zero RPC calls.

    Uses the static cache (type_id + bounding_box) built once at scene start,
    combined with per-tick transform/velocity from the already-fetched snapshot.

    Parameters
    ----------
    static_cache   : dict[int, dict] — from build_actor_static_cache()
    world_snapshot : carla.WorldSnapshot — already fetched this tick

    Returns
    -------
    dict[int, dict]  keyed by actor ID
    """
    result = {}
    for actor_id, static in static_cache.items():
        actor_snap = world_snapshot.find(actor_id)
        if actor_snap is None:
            continue
        try:
            t = actor_snap.get_transform()
            v = actor_snap.get_velocity()
            result[actor_id] = {
                'type_id':      static['type_id'],
                'bounding_box': static['bounding_box'],
                'transform': {
                    'x': t.location.x, 'y': t.location.y, 'z': t.location.z,
                    'pitch': t.rotation.pitch, 'yaw': t.rotation.yaw, 'roll': t.rotation.roll,
                    'matrix': [list(row) for row in t.get_matrix()],
                },
                'velocity': {'x': v.x, 'y': v.y, 'z': v.z},
            }
        except Exception:
            pass
    return result


def save_ego_pose(transform, timestamp, ego_pose_dir):
    """Save the ego vehicle's pose. transform is a plain Python dict."""
    pose = {
        "timestamp": timestamp,
        "translation": {
            "x": transform['x'], "y": transform['y'], "z": transform['z']
        },
        "rotation": {
            "pitch": transform['pitch'], "yaw": transform['yaw'], "roll": transform['roll']
        }
    }
    pose_path = os.path.join(ego_pose_dir, f"{timestamp}.json")
    with open(pose_path, 'w') as f:
        json.dump(pose, f, separators=(',', ':'))


def generate_map_mask(base_save_path):
    """Generate map mask using the external generate_map_mask.py script."""
    print("\n" + "="*60)
    print("GENERATING MAP MASK BEFORE SENSOR DATA COLLECTION")
    print("="*60)
    try:
        script = Path(__file__).parent / 'generate_map_mask.py'
        if not script.exists():
            print("generate_map_mask.py not found in current directory")
            return False

        cmd = [
            sys.executable,
            str(script),
            '--output', base_save_path,
            '--resolution', 'medium',
            '--altitude', '400.0',
            '--target-resolution', '0.1'
        ]
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=600, cwd=os.getcwd())
        if result.returncode == 0:
            files = (glob.glob(os.path.join(base_save_path, '*.png')) +
                     glob.glob(os.path.join(base_save_path, '*.json')))
            if files:
                print(f"Created map files: {[os.path.basename(f) for f in files]}")
                return True
            print("Warning: map generation reported success but no map files found")
            return False
        print("Map mask generation failed!")
        print("STDERR:", result.stderr)
        return False
    except subprocess.TimeoutExpired:
        print("Map mask generation timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"Error running map mask generation: {e}")
        return False
    finally:
        print("="*60)


def collect_log_info(world, ego_vehicle, base_save_path):
    """Collects and saves simulation log info for NuScenes log.json generation."""
    map_name = world.get_map().name.split('/')[-1]
    vehicle_blueprint = ego_vehicle.type_id
    date_captured = time.strftime("%Y-%m-%d")
    logfile = f"carla_log_{date_captured}.log"
    log_info = {
        "logfile": logfile,
        "vehicle": vehicle_blueprint,
        "date_captured": date_captured,
        "location": map_name,
        "start_time_unix": time.time()
    }
    log_info_path = os.path.join(base_save_path, LOG_INFO_FILENAME)
    with open(log_info_path, 'w') as f:
        json.dump(log_info, f, indent=2)


def main():
    """Initialise CARLA, configure sensors, and run data collection."""
    client = None
    vehicle_list = []

    try:
        with open(ROOT / 'config.yml', 'r') as f:
            config = yaml.safe_load(f)

        # Inject instance cameras where collect_bbox is true on an RGB camera.
        config["sensors"] = process_sensor_config(config["sensors"])

        sim_config     = config["simulation"]
        sensors_config = config["sensors"]
        traffic_config = sim_config.get("traffic", {})

        frequency_hz = int(sim_config.get("frequency_hz", 20))
        if frequency_hz <= 0:
            raise ValueError("simulation.frequency_hz must be a positive integer")

        num_scenes       = sim_config["num_scenes"]
        seconds_per_scene = sim_config["seconds_per_scene"]
        ticks_per_scene  = int(seconds_per_scene * frequency_hz)
        base_save_path   = sim_config["base_save_path"]

        # Build a static blueprint_id map so listeners never re-read config.yml
        blueprint_id_map = {s["name"]: s["blueprint"] for s in sensors_config}

        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)

        world    = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode   = True
        settings.fixed_delta_seconds = 1.0 / float(frequency_hz)
        settings.substepping         = False
        world.apply_settings(settings)

        # Cache static world vehicles once — expensive RPC, never repeat per frame.
        static_vehicles = get_static_vehicle_env_objects(world)
        print(f"Found {len(static_vehicles)} static world vehicles for bbox export.")

        os.makedirs(base_save_path, exist_ok=True)

        print("Setting up traffic...")
        vehicle_list = setup_traffic(client, world, traffic_config)

        print("Letting traffic settle...")
        settle_ticks = max(1, int(round(2.5 * frequency_hz)))
        for _ in range(settle_ticks):
            world.tick()

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        scene_paths = []
        log_info_collected = False

        first_scene_id = _next_scene_id(base_save_path)
        if first_scene_id > 1:
            print(f"Existing scenes detected — new scenes will start at scene_{first_scene_id}")

        for scene_id in range(first_scene_id, first_scene_id + num_scenes):
            scene_completed = False
            max_scene_retries = 3

            for scene_retry in range(max_scene_retries):
                sensor_list = []
                vehicle     = None

                # One executor per scene attempt.  shutdown(wait=True) at the end
                # of the scene guarantees all file writes are flushed before cleanup.
                executor = ThreadPoolExecutor(max_workers=IO_WORKERS)

                # Two queues:
                #   raw_queue  — filled by thin sensor callbacks (CARLA thread)
                #   done_queue — filled by workers when a frame write finishes
                raw_queue  = Queue()
                done_queue = Queue()

                try:
                    sensor_names = [s["name"] for s in sensors_config]
                    save_path    = create_scene_folders(scene_id, sensor_names, base_save_path)
                    scene_paths.append(save_path)
                    blueprint_library = world.get_blueprint_library()

                    print(f"\nScene {scene_id} - Attempt {scene_retry + 1}/{max_scene_retries}")
                    try:
                        vehicle = spawn_ego_vehicle(world, blueprint_library, traffic_manager)
                    except RuntimeError as e:
                        print(f"Failed to spawn ego vehicle: {e}")
                        if scene_retry < max_scene_retries - 1:
                            print("Retrying scene...")
                            continue
                        else:
                            print("Max retries reached, skipping scene")
                            break

                    print("Ego vehicle spawned, stabilising...")
                    stabilize_ticks = max(1, int(round(1.0 * frequency_hz)))
                    for _ in range(stabilize_ticks):
                        world.tick()

                    if not log_info_collected:
                        collect_log_info(world, vehicle, base_save_path)
                        log_info_collected = True

                    ego_pose_dir = os.path.join(save_path, EGO_POSE_FOLDER)
                    os.makedirs(ego_pose_dir, exist_ok=True)  # once, not per tick

                    # --------------------------------------------------------
                    # Register sensors with thin callbacks.
                    # blueprint_id is captured at registration time — the callback
                    # never opens config.yml.
                    # actor_snapshot / ego_transform are mutable containers so
                    # the tick loop can update them in-place and the callback
                    # closure always sees the latest value.
                    # --------------------------------------------------------
                    # We use a list-of-one as a mutable reference so the lambda
                    # captures the container, not a frozen value.
                    snapshot_ref    = [{}]   # snapshot_ref[0] = current actor snapshot
                    ego_tf_ref      = [None] # ego_tf_ref[0]   = current ego transform

                    for sensor_cfg in sensors_config:
                        bp_sensor = blueprint_library.find(sensor_cfg["blueprint"])
                        for attr, value in sensor_cfg["attributes"].items():
                            bp_sensor.set_attribute(attr, value)
                        loc = sensor_cfg["transform"]["location"]
                        rot = sensor_cfg["transform"]["rotation"]
                        transform = carla.Transform(
                            carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
                            carla.Rotation(pitch=rot.get("pitch", 0),
                                           yaw=rot["yaw"],
                                           roll=rot.get("roll", 0))
                        )
                        actor = world.spawn_actor(bp_sensor, transform, attach_to=vehicle)
                        sensor_list.append(actor)

                        _name    = sensor_cfg["name"]
                        _bp_id   = blueprint_id_map[_name]
                        actor.listen(
                            lambda data,
                                   q=raw_queue,
                                   name=_name,
                                   path=save_path,
                                   bp=_bp_id,
                                   snap_ref=snapshot_ref,
                                   etf_ref=ego_tf_ref:
                            sensor_callback(data, q, name, path, bp,
                                            snap_ref[0], etf_ref[0])
                        )

                    # --------------------------------------------------------
                    # Warm-up ticks — discard data, let sensors stabilise.
                    # --------------------------------------------------------
                    WARMUP_TICKS = 10
                    print(f"Running {WARMUP_TICKS} warm-up ticks...")
                    for _ in range(WARMUP_TICKS):
                        world.tick()
                        # Drain any warm-up frames so the queue stays empty
                        while not raw_queue.empty():
                            try:
                                raw_queue.get_nowait()
                            except Empty:
                                break

                    # --------------------------------------------------------
                    # Main collection tick loop.
                    # Per tick:
                    #   1. world.tick()  — advance simulation, callbacks fire
                    #   2. Build actor snapshot from world_snapshot (zero RPC)
                    #   3. Save ego pose (main thread, blocking but tiny)
                    #   4. Drain raw_queue — submit each frame to the executor
                    # --------------------------------------------------------
                    print(f"\nStarting data collection for scene {scene_id}...")
                    print(f"Expected frames: {ticks_per_scene} "
                          f"({frequency_hz}Hz × {seconds_per_scene}s)")

                    # Cache static actor metadata once — type_id + bounding_box
                    # never change, so we avoid world.get_actors() every tick.
                    actor_static_cache = build_actor_static_cache(world, vehicle.id)
                    print(f"Actor cache built: {len(actor_static_cache)} NPC actors")

                    pending_futures = []

                    for tick in range(ticks_per_scene):
                        # --------------------------------------------------------
                        # Backpressure: if too many futures are pending, wait for
                        # the oldest ones to finish before advancing the simulation.
                        # This caps memory usage and prevents OOM crashes.
                        # --------------------------------------------------------
                        if len(pending_futures) >= MAX_PENDING_FUTURES:
                            cutoff = len(pending_futures) // 2
                            for f in pending_futures[:cutoff]:
                                try:
                                    f.result(timeout=10.0)
                                except Exception:
                                    pass
                            pending_futures = [f for f in pending_futures[cutoff:]
                                               if not f.done()]

                        world.tick()

                        world_snap = world.get_snapshot()
                        timestamp = int(world_snap.timestamp.elapsed_seconds * 1000)

                        # Build actor snapshot from world_snapshot — zero RPC calls.
                        actor_snap    = build_actor_snapshot(actor_static_cache, world_snap)
                        # Ego transform: use world snapshot (zero RPC).
                        ego_snap = world_snap.find(vehicle.id)
                        if ego_snap is not None:
                            _et = ego_snap.get_transform()
                        else:
                            _et = vehicle.get_transform()
                        # Serialize to plain Python — no carla C++ object crosses thread boundary
                        ego_transform = {
                            'x': _et.location.x, 'y': _et.location.y, 'z': _et.location.z,
                            'pitch': _et.rotation.pitch, 'yaw': _et.rotation.yaw, 'roll': _et.rotation.roll,
                            'matrix': [list(row) for row in _et.get_matrix()],
                        }

                        # Update shared references so the NEXT callback invocations
                        # (which may not have fired yet) see the fresh snapshot.
                        # Callbacks that already fired this tick already captured
                        # their snapshot via the closure at call time.
                        snapshot_ref[0] = actor_snap
                        ego_tf_ref[0]   = ego_transform

                        # Ego pose — directory already created above.
                        save_ego_pose(ego_transform, timestamp, ego_pose_dir)

                        # Drain raw sensor queue — submit writes to executor.
                        drained = 0
                        while not raw_queue.empty():
                            try:
                                item = raw_queue.get_nowait()
                            except Empty:
                                break
                            (s_payload, s_ts, s_name, s_path,
                             s_bp, s_snap, s_etf) = item
                            f = executor.submit(
                                write_sensor_data,
                                s_payload, s_ts, s_name, s_path, s_bp,
                                s_snap, s_etf,
                                static_vehicles, done_queue
                            )
                            pending_futures.append(f)
                            drained += 1

                        if tick % 10 == 0:
                            pending_futures = [f for f in pending_futures if not f.done()]
                            print(f"  Tick {tick + 1}/{ticks_per_scene} "
                                  f"— queued {drained} frames, "
                                  f"{len(pending_futures)} futures pending")

                    # --------------------------------------------------------
                    # Scene done — wait for all worker writes to finish.
                    # --------------------------------------------------------
                    print("Flushing I/O workers...")
                    executor.shutdown(wait=True)
                    print(f"Scene {scene_id} complete — all frames written.")

                    scene_completed = True
                    break  # exit retry loop

                except Exception as e:
                    import traceback
                    print(f"Error during scene {scene_id}: {e}")
                    traceback.print_exc()
                    # Shut executor down cleanly even on error
                    try:
                        executor.shutdown(wait=False)
                    except Exception:
                        pass
                    if scene_retry < max_scene_retries - 1:
                        print("Retrying scene...")
                    else:
                        print("Max retries reached, skipping scene")

                finally:
                    # ----------------------------------------------------------
                    # Cleanup order matters:
                    #   1. Stop all sensor listeners (no more callbacks fire).
                    #   2. Tick once so CARLA server processes the stop commands
                    #      and flushes any in-flight callbacks.
                    #   3. Drain the raw_queue so no orphaned items remain.
                    #   4. Wait for executor workers to finish (they may hold refs
                    #      to CARLA data objects — don't destroy actors beneath them).
                    #   5. Destroy sensors, tick, destroy vehicle, tick.
                    #      Two ticks give the server time to GC each batch.
                    # ----------------------------------------------------------

                    # Step 1 — stop listeners.
                    for sensor in sensor_list:
                        try:
                            if sensor.is_alive:
                                sensor.stop()
                        except Exception:
                            pass

                    # Step 2 — tick to flush in-flight callbacks.
                    try:
                        world.tick()
                    except Exception:
                        pass

                    # Step 3 — drain raw_queue (discard; scene is ending).
                    while not raw_queue.empty():
                        try:
                            raw_queue.get_nowait()
                        except Exception:
                            break

                    # Step 4 — wait for all I/O workers to finish.
                    try:
                        executor.shutdown(wait=True)
                    except Exception:
                        pass

                    # Step 5a — destroy sensors.
                    for sensor in sensor_list:
                        try:
                            if sensor.is_alive:
                                sensor.destroy()
                        except Exception:
                            pass
                    sensor_list.clear()

                    # Tick so server processes sensor destructions.
                    try:
                        world.tick()
                    except Exception:
                        pass

                    # Step 5b — destroy ego vehicle.
                    if vehicle is not None:
                        try:
                            if vehicle.is_alive:
                                vehicle.destroy()
                        except Exception:
                            pass
                        vehicle = None

                    # Tick so server processes vehicle destruction.
                    try:
                        world.tick()
                    except Exception:
                        pass

            if not scene_completed:
                print(f"Failed to complete scene {scene_id} after {max_scene_retries} attempts")
            else:
                # Let the server settle between scenes before spawning new actors.
                print("Letting server settle before next scene...")
                settle_between = max(1, int(round(1.0 * frequency_hz)))
                for _ in range(settle_between):
                    try:
                        world.tick()
                    except Exception:
                        break

        # ------------------------------------------------------------------
        # Post-collection: clean up partial frames, then run 2D bbox annotation
        # (only for cameras with collect_bbox: true — default is false).
        # ------------------------------------------------------------------
        for path in scene_paths:
            print(f"Cleaning scene data: {path}")
            sensors_for_cleanup = list(sensor_names) + [EGO_POSE_FOLDER]
            clean_scene_data(path, sensors_for_cleanup)
        print("All scenes cleaned.")

        for path in scene_paths:
            process_scene(path)

    except KeyboardInterrupt:
        print(" - Interrupted by user.")
    except Exception as e:
        print(f"Fatal error during initialisation: {e}")
        import traceback; traceback.print_exc()

    finally:
        # Destroy traffic actors.
        if vehicle_list:
            print('\nDestroying traffic...')
            try:
                client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])
            except Exception as e:
                print(f"Warning: Failed to destroy some traffic actors: {e}")
            time.sleep(0.5)

        # Map mask: use committed asset if available, otherwise generate and commit.
        try:
            if 'base_save_path' in locals() and base_save_path:
                maps_dir = ROOT / 'maps'
                maps_dir.mkdir(exist_ok=True)

                # Check whether a committed basemap already exists for any map.
                import glob as _glob
                committed = list(maps_dir.glob('*_basemap.png'))
                if committed:
                    print(f"\nCommitted map asset found ({committed[0].name}) — skipping map mask generation.")
                    print("  Delete or rename maps/*_basemap.png to force regeneration.")
                else:
                    print(f"\nNo committed map asset found — generating map mask...")
                    success = generate_map_mask(base_save_path)
                    if not success:
                        print("WARNING: Map mask generation failed.")
                        print("  Re-run collection while CARLA is running, or manually run:")
                        print(f"  python collection/generate_map_mask.py --output {base_save_path}")
                    else:
                        # Copy the generated files into maps/ so they can be committed.
                        generated = _glob.glob(os.path.join(base_save_path, '*_basemap.png'))
                        generated_json = _glob.glob(os.path.join(base_save_path, '*.json'))
                        # Only copy the map metadata JSON (not log_info.json).
                        map_jsons = [
                            p for p in generated_json
                            if os.path.basename(p) != LOG_INFO_FILENAME
                            and '_basemap' not in os.path.basename(p)
                        ]
                        copied = []
                        for src in generated + map_jsons:
                            dst = maps_dir / os.path.basename(src)
                            import shutil as _shutil
                            _shutil.copy2(src, dst)
                            copied.append(dst.name)
                        if copied:
                            print(f"Map assets copied to maps/: {copied}")
                            print("  Commit the maps/ directory to avoid regeneration on future runs.")
                        else:
                            print("WARNING: generate_map_mask reported success but no map files found.")
                            print(f"  Expected files matching {base_save_path}/*_basemap.png")
        except Exception as e:
            print(f"Error during map mask handling: {e}")


if __name__ == "__main__":
    main()

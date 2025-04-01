import random
import time
import carla

def setup_traffic(client, world, traffic_config):
    """Configure and spawn traffic (vehicles and pedestrians)"""
    try:
        # Get blueprints for vehicles and pedestrians
        blueprints = world.get_blueprint_library().filter('vehicle.*')
        if traffic_config.get("safe_spawn", True):
            blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']
        blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')

        # Get spawn points for vehicles
        spawn_points = world.get_map().get_spawn_points()
        num_spawn_points = len(spawn_points)

        # Get numbers from config (remove default values)
        num_vehicles = traffic_config["num_vehicles"] 
        num_pedestrians = traffic_config["num_pedestrians"]  
        
        # Limit vehicles to available spawn points
        if num_vehicles > num_spawn_points:
            print(f"Warning: Requested {num_vehicles} vehicles but only {num_spawn_points} spawn points available")
            num_vehicles = num_spawn_points

        # -------------
        # Spawn Vehicles
        # -------------
        batch = []
        vehicle_list = []
        random.shuffle(spawn_points)
        for n, transform in enumerate(spawn_points):
            if n >= num_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(carla.command.SpawnActor(blueprint, transform)
                .then(carla.command.SetAutopilot(carla.command.FutureActor, True)))

        # Apply batch spawn for vehicles
        for response in client.apply_batch_sync(batch, True):
            if response.error:
                print(f"Error spawning vehicle: {response.error}")
            else:
                vehicle_list.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # Some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        
        # 1. Take all the random locations to spawn
        spawn_points = []
        for i in range(num_pedestrians):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_point.location.z += 1  # Spawn slightly above ground
                spawn_points.append(spawn_point)

        # 2. Spawn the walker objects
        batch = []
        walker_speed = []
        walkers_list = []
        
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # Set the max speed
            if walker_bp.has_attribute('speed'):
                if random.random() > percentagePedestriansRunning:
                    # Walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # Running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                walker_speed.append(0.0)
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

        # 2.1 Apply batch spawn
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if not results[i].error:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2

        # 3. Spawn the walker controllers
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))

        # 3.1 Apply batch spawn
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if not results[i].error:
                walkers_list[i]["con"] = results[i].actor_id

        # 4. Initialize each controller and set target to walk to
        walker_controller_list = []
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(len(walkers_list)):
            controller_id = walkers_list[i]["con"]
            controller = world.get_actor(controller_id)
            if controller is None:
                continue
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(float(walker_speed[i]))
            walker_controller_list.append(controller_id)

        # Set up traffic manager for vehicles
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.global_percentage_speed_difference(30.0)

        # Set vehicle lights if specified
        if traffic_config.get("car_lights_on", False):
            for vehicle_id in vehicle_list:
                vehicle = world.get_actor(vehicle_id)
                if vehicle is not None:
                    vehicle.set_light_state(carla.VehicleLightState.All)

        all_ids = vehicle_list + [w["id"] for w in walkers_list] + walker_controller_list
        print(f"Successfully spawned {len(vehicle_list)} vehicles and {len(walkers_list)} walkers")
        return all_ids

    except Exception as e:
        print(f"Error setting up traffic: {e}")
        return []

def spawn_ego_vehicle(world, blueprint_library, traffic_manager, max_retries=10):
    """Safely spawn the ego vehicle by trying different spawn points"""
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)  # Randomize spawn points
    
    bp = blueprint_library.find('vehicle.lincoln.mkz')
    bp.set_attribute('role_name', 'ego')
    
    for retry in range(max_retries):
        if retry > 0:
            print(f"Retrying ego vehicle spawn (attempt {retry + 1}/{max_retries})")
        
        # Try each spawn point until we find one that works
        for spawn_point in spawn_points:
            try:
                vehicle = world.try_spawn_actor(bp, spawn_point)
                if vehicle is not None:
                    # Successfully spawned - set autopilot and return
                    vehicle.set_autopilot(True, traffic_manager.get_port())
                    print(f"Successfully spawned ego vehicle after {retry + 1} attempts")
                    return vehicle
            except Exception as e:
                continue
        
        # If we get here, no spawn points worked - wait a bit and try again
        print("All spawn points blocked, waiting for clearance...")
        world.tick()  # Tick the world to update physics
        time.sleep(0.5)
    
    raise RuntimeError(f"Failed to spawn ego vehicle after {max_retries} attempts")
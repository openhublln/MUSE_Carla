import carla
import argparse
import time

def list_available_maps(client):
    """List all available maps."""
    print("\nAvailable maps:")
    for map_name in client.get_available_maps():
        # Convert full path to just the map name
        map_name = map_name.split('/')[-1]
        current = " (current)" if map_name == client.get_world().get_map().name.split('/')[-1] else ""
        print(f"- {map_name}{current}")

def change_map(client, map_name):
    """Change to the specified map."""
    print(f"\nChanging to map: {map_name}")
    try:
        # Store current timeout and set a longer one for map loading
        original_timeout = 20.0  # Default timeout
        client.set_timeout(60.0)  # Increase timeout to 60 seconds for map loading
        
        try:
            # Load the new world
            print("Loading world (this may take a while)...")
            world = client.load_world(map_name)
            
            # Wait for the world to be ready
            max_tries = 30
            for i in range(max_tries):
                try:
                    # Check if the new world is ready
                    if world.get_map().name.split('/')[-1] == map_name:
                        print(f"Map loaded successfully after {i+1} attempts")
                        break
                except:
                    if i == max_tries - 1:
                        print("Warning: Could not verify map loading")
                    time.sleep(1)
                    continue
            
            # Restore synchronous mode setting
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
            
            # Let the server and client settle
            time.sleep(2.0)
            
            # Verify the map change
            current_map = world.get_map().name.split('/')[-1]
            if current_map == map_name:
                print(f"Successfully verified map change to: {current_map}")
                return True
            else:
                print(f"Warning: Map verification failed. Current map: {current_map}")
                return False
            
        finally:
            # Restore original timeout
            client.set_timeout(original_timeout)
        
    except Exception as e:
        print(f"Error changing map: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='CARLA Map Changer')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA host server IP')
    parser.add_argument('--port', default=2000, type=int, help='CARLA host server port')
    parser.add_argument('--map', help='Name of the map to load (if not specified, will just list available maps)')
    parser.add_argument('--timeout', default=20.0, type=float, help='Timeout in seconds for the client connection')
    
    args = parser.parse_args()
    
    try:
        # Connect to CARLA
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        
        # Get current world info
        world = client.get_world()
        current_map = world.get_map().name.split('/')[-1]
        print(f"\nCurrently loaded map: {current_map}")
        
        # List available maps
        list_available_maps(client)
        
        # If a map was specified, try to change to it
        if args.map:
            if args.map == current_map:
                print(f"\nMap '{args.map}' is already loaded!")
            else:
                success = change_map(client, args.map)
                if success:
                    print("\nMap changed successfully!")
                else:
                    print("\nFailed to change map!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())

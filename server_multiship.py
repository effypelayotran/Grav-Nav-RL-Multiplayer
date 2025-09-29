# Import required libraries for async operations, unique IDs, numerical computing, ML, and web framework
import asyncio  # For asynchronous programming (non-blocking operations)
import uuid  # For generating unique identifiers for each client
import numpy as np  # For numerical computations (physics calculations)
import torch  # For machine learning model operations
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # Web framework for real-time communication
from fastapi.middleware.cors import CORSMiddleware  # Allow cross-origin requests from frontend
from stable_baselines3 import PPO  # Pre-trained reinforcement learning model
from environment import MultiShipOrbitalEnvironment  # Custom physics environment

######################## INITIALIZE WEBAPP & WEBSOCKET CONNECTIONS ########################
# Create a lock to prevent race conditions when multiple clients modify shared data simultaneously
clients_lock = asyncio.Lock()

# Initialize the FastAPI web application
app = FastAPI()

# Configure CORS to allow the frontend (running on different port) to connect to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (in production, specify exact frontend URL)
    allow_credentials=True,  # Allow cookies/authentication
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

######################## INITIALIZE PHYSICS ENVIRONMENT ########################
# Initialize the shared physics environment that all ships exist in
env = MultiShipOrbitalEnvironment(dt=1.0/60.0)

# Dictionary to store all connected clients and their metadata
# Key: client_id (UUID), Value: dict with websocket, type, ship_id, and model info
clients = {}  # client_id: {"ws": ws, "type": type, "ship_id": ship_id, "model": model}

# Path to the pre-trained PPO model file (baseline AI ship uses this)
model_path = './models/model_22-51-24-_18-05-2025/ppo_orbital_model.zip'

# Global flags to control the simulation state
simulation_running = False  # Whether the physics simulation is currently running
simulation_task = None  # Reference to the async task running the simulation loop
baseline_ship_id = "baseline_ship"  # Special ID for the AI-controlled baseline ship
baseline_model_loaded = None  # The loaded PPO model for the baseline ship

# Tick-based authoritative networking
current_tick = 0  # Current authoritative simulation tick
tick_rate = 60  # Simulation ticks per second

# Lockstep simulation - wait for all clients before advancing
pending_actions = {}  # client_id: action for current tick
action_timeout = 0.1  # 100ms timeout for client actions (10 ticks at 60Hz)
last_action_request_time = 0  # When we last requested actions from clients
state_send_time = 0  # When we last sent state update to clients

# Store historical trail data for each ship (so new clients can see past movement)
trail_history = {}  # ship_id: [(x, y, timestamp), ...]


######################## LEADERBOARD SYSTEM ########################
# Leaderboard system to track ship survival times
leaderboard = {}  # ship_id: {"name": name, "steps": steps, "start_time": time, "alive": True}

def add_to_leaderboard(ship_id, name="Unknown"):
    """Add a new ship to the global leaderboard tracking system"""
    global leaderboard
    leaderboard[ship_id] = {
        "name": name,  # Display name for the ship
        "steps": 0,  # Current survival time in simulation steps
        "start_time": asyncio.get_event_loop().time(),  # When the ship was created
        "alive": True  # Whether the ship is still active
    }

def remove_from_leaderboard(ship_id):
    """Completely remove a ship from the global leaderboard"""
    global leaderboard
    if ship_id in leaderboard:
        del leaderboard[ship_id]

def update_leaderboard_steps(ship_id, steps):
    """Update the survival step count for a ship"""
    global leaderboard
    if ship_id in leaderboard:
        leaderboard[ship_id]["steps"] = steps

def get_top_leaderboard():
    """Get the top 10 ships ranked by survival time (steps)"""
    global leaderboard
    # Sort all ships by their step count in descending order (longest survivors first)
    sorted_ships = sorted(leaderboard.items(), key=lambda x: x[1]["steps"], reverse=True)
    return sorted_ships[:10]  # Return only the top 10


######################## MESSAGE FORMAT HELPERS ########################
def create_message(message_type, payload=None, tick=None, client_id=None):
    """Create a standardized message with header and payload"""
    message = {
        "header": {
            "version": "1.0",
            "type": message_type,
            "tick": tick if tick is not None else current_tick,
            "timestamp": asyncio.get_event_loop().time(),
            "client_id": client_id
        },
        "payload": payload or {}
    }
    return message

def validate_message_tick(message_tick, ship_id):
    """Validate if a message tick is acceptable (current or future only)"""
    global current_tick
    
    # Accept current tick
    if message_tick == current_tick:
        return True, "current"
    
    # Accept future ticks (client prediction)
    if message_tick > current_tick:
        return True, "future"
    
    # Reject old ticks (too late)
    return False, "too_old"

def get_active_clients():
    """Get list of client IDs that need to provide actions (manual control clients only)"""
    active_clients = []
    for client_id, client in clients.items():
        # Only manual control clients need to provide actions
        # Model clients are handled server-side
        if client['type'] == 'manual' and client.get('ship_id'):
            active_clients.append(client_id)
    return active_clients

def all_actions_received():
    """Check if all active clients have provided actions for current tick"""
    active_clients = get_active_clients()
    
    # If no active clients, we can proceed immediately
    if len(active_clients) == 0:
        return True
    
    # Check if all active clients have provided actions
    for client_id in active_clients:
        if client_id not in pending_actions:
            return False
    
    return True

async def request_actions_from_clients():
    """Send action request to all active clients"""
    global last_action_request_time
    last_action_request_time = asyncio.get_event_loop().time()
    
    active_clients = get_active_clients()
    for client_id in active_clients:
        client = clients[client_id]
        try:
            action_request = create_message(
                "action_request",
                {
                    "tick": current_tick,
                    "ship_id": client['ship_id']
                },
                client_id=client_id
            )
            await client["ws"].send_json(action_request)
        except Exception as e:
            print(f"Error sending action request to client {client_id}: {e}")
    
    # Print when we send action requests
    if len(active_clients) > 0:
        print(f"[TICK {current_tick}] Action request sent at {last_action_request_time:.6f}")

def clear_pending_actions():
    """Clear all pending actions for next tick"""
    global pending_actions
    pending_actions.clear()

######################## HELPER FUNCTIONS ########################
# Helper function to convert raw ship state (x,y,vx,vy) into the 7-dimensional observation
# that the AI model expects (this matches the run_and_view_episode.py training environment)
def convert_state(state, env, ship):
    # Extract position and velocity components
    x, y, vx, vy = state 
    
    # Calculate distance from center (central mass)
    r = np.sqrt(x**2 + y**2)
    r = max(r, 1e-5)  
    
    # Calculate radial and tangential velocity components
    v_radial = (x * vx + y * vy) / r  # Velocity toward/away from center
    v_tangential = (x * vy - y * vx) / r  # Velocity perpendicular to radial direction

    initial_r = ship['init_r']  # Initial orbital radius
    
    # Flag indicating if ship is very close to target orbit (radius 1.0)
    flag = 1.0 if np.abs(r - 1.0) < 0.01 else 0.0
    
    # Calculate specific energy (kinetic + potential)
    specific_energy = 0.5 * (vx**2 + vy**2) - env.GM / r
    
    # Calculate angular momentum
    angular_momentum = r * v_tangential
    
    # Calculate radial error from target orbit
    r_err = r - 1.0
    r_max_err = max(abs(initial_r - 1), 1e-2)  # Maximum expected error
    scaled_r_err = np.clip((r_err / r_max_err) * 2, -2, 2)  # Scale and clip error
    
    # Return 7-dimensional observation vector for the AI model
    obs = np.array([
        scaled_r_err,      # How far from target orbit (scaled)
        v_radial,          # Radial velocity component
        v_tangential,      # Tangential velocity component
        1 - initial_r,     # Initial orbital error
        flag,              # Whether at target orbit
        specific_energy,   # Total energy
        angular_momentum   # Angular momentum
    ], dtype=np.float32)
    return obs

def add_baseline_ship():
    """Create and add the baseline AI-controlled ship to the simulation"""
    global baseline_model_loaded
    if baseline_ship_id not in env.ships:  
        # Add ship at fixed radius 1.5
        env.add_ship(baseline_ship_id, r0=1.5)  
        # Load the pre-trained PPO model for the baseline ship
        baseline_model_loaded = PPO.load(model_path)
        # Add to leaderboard with a recognizable name
        add_to_leaderboard(baseline_ship_id, "Baseline AI")

        print("Baseline ship added to environment and leaderboard")

def remove_baseline_ship():
    """Remove the baseline ship from the simulation"""
    if baseline_ship_id in env.ships:
        env.remove_ship(baseline_ship_id)  # Remove from physics environment
        remove_from_leaderboard(baseline_ship_id)  # Remove from leaderboard
        print("Baseline ship removed from environment and leaderboard")


######################## MAIN PHYSICS SIMULATION LOOP ########################
# The main physics simulation loop that runs independently of client connections

async def global_simulation_loop():
    """Global simulation loop that runs lockstep - waits for all client actions before advancing"""
    global simulation_running, trail_history, current_tick, pending_actions
    
    while simulation_running:  # Continue until simulation is stopped
        current_tick += 1  # Increment authoritative tick
        clear_pending_actions()  # Clear actions from previous tick
        
        async with clients_lock:  # Prevent race conditions with client operations
            if env.ships:  # Only run if there are ships in the environment
                # Only request actions if there are active clients
                active_clients = get_active_clients()
                total_ships = len(env.ships)
                print(f"[TICK {current_tick}] Ships in environment: {total_ships}, Active clients (manual): {len(active_clients)}")
                
                if len(active_clients) > 0:
                    # Request actions from all active clients
                    await request_actions_from_clients()
                    
                    # Wait for all actions or timeout
                    start_wait_time = asyncio.get_event_loop().time()
                    while not all_actions_received():
                        current_time = asyncio.get_event_loop().time()
                        if current_time - start_wait_time > action_timeout:
                            print(f"Timeout waiting for actions at tick {current_tick}")
                            break
                        await asyncio.sleep(0.001)  # Small delay to prevent busy waiting
                # else:
                #     # print(f"No active clients at tick {current_tick}, proceeding with baseline ship only")
                
                # Dictionary to collect all ship actions for this physics step
                actions = {}
                
                # Initialize baseline variables
                baseline_action = 0.0
                baseline_init_r = 1.5  # Default initial radius for baseline ship
                
                # Handle baseline ship (always uses the pre-trained PPO model)
                if baseline_ship_id in env.ships and not env.ships[baseline_ship_id]['done']:
                    ship = env.ships[baseline_ship_id]  # Get baseline ship state
                    # Convert ship state to numpy array for model input
                    state = np.array([ship['x'], ship['y'], ship['vx'], ship['vy']], dtype=np.float32)
                    # Convert to 7-dimensional observation for the AI model
                    obs = convert_state(state, env, ship)
                    # Convert to PyTorch tensor and add batch dimension
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                    # Use the model to predict action (no gradient computation needed)
                    with torch.no_grad():
                        action, _ = baseline_model_loaded.predict(obs_tensor, deterministic=True)
                    # Store the predicted tangential thrust action
                    action[0] = 0.0
                    actions[baseline_ship_id] = float(action[0])
                    
                    # Store baseline action for broadcasting
                    baseline_action = float(action[0])
                    # Store baseline initial radius for broadcasting
                    baseline_init_r = ship.get('init_r', 1.5)
                
                # Handle manual client-controlled ships using their provided actions
                for client_id, action in pending_actions.items():
                    client = clients.get(client_id)
                    if client and client.get('ship_id') and client['type'] == 'manual':
                        ship_id = client['ship_id']
                        if ship_id in env.ships and not env.ships[ship_id]['done']:
                            actions[ship_id] = action
                
                # Handle model client-controlled ships (server-side prediction)
                for client_id, client in clients.items():
                    if client['type'] == 'model' and client.get('model_loaded', False) and client.get('ship_id'):
                        ship_id = client['ship_id']
                        if ship_id in env.ships and not env.ships[ship_id]['done']:
                            ship = env.ships[ship_id]
                            
                            # Convert ship state to observation for the AI model (same as baseline)
                            state = np.array([ship['x'], ship['y'], ship['vx'], ship['vy']], dtype=np.float32)
                            obs = convert_state(state, env, ship)
                            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                            
                            # Use the client's uploaded model to predict action
                            with torch.no_grad():
                                action, _ = client['model'].predict(obs_tensor, deterministic=True)
                            
                            actions[ship_id] = float(action[0])
                            # Store the action in client for broadcasting
                            client['last_action'] = float(action[0])
                            print(f"Model client {client_id} action: {float(action[0]):.3f}")
                
                # Step the physics environment forward by one timestep with all actions
                env.step(actions)
                
                # Update leaderboard steps for all alive ships
                for ship_id, ship in env.ships.items():
                    if not ship['done']:
                        update_leaderboard_steps(ship_id, ship.get('steps', 0))
                    else:
                        # Ship died, completely remove from leaderboard
                        remove_from_leaderboard(ship_id)
                
                # Update trail history for all ships (for visualization)
                current_time = asyncio.get_event_loop().time()
                for ship_id, ship in env.ships.items():
                    if ship_id not in trail_history:
                        trail_history[ship_id] = []  # Initialize trail for new ship
                    # Add current position to trail
                    trail_history[ship_id].append((ship['x'], ship['y'], current_time))
                    # Keep only last 1000 points to prevent memory issues
                    if len(trail_history[ship_id]) > 1000:
                        trail_history[ship_id] = trail_history[ship_id][-1000:]
                
                # Get current state of all ships for broadcasting to clients
                states = env.get_states()
                
                # Get current top 10 leaderboard
                top_leaderboard = get_top_leaderboard()
                
                # Broadcast current state to all connected clients
                global state_send_time
                state_send_time = asyncio.get_event_loop().time()
                to_remove = []  # Track clients that need to be removed
                for client_id, client in list(clients.items()):
                    try:
                        # Prepare payload with client-specific data
                        payload = {
                            "ships": states,  # Current state of all ships
                            "your_ship_id": client.get('ship_id'),  # This client's ship ID
                            "trail_history": trail_history,  # Historical trail data
                            "leaderboard": top_leaderboard,  # Current leaderboard
                            "baseline_action": baseline_action,  # Add baseline action
                            "baseline_init_r": baseline_init_r  # Add baseline initial radius
                        }
                        
                        # Add model action if this is a model client
                        if client['type'] == 'model' and client.get('model_loaded', False):
                            payload["model_action"] = client.get('last_action', 0.0)
                        
                        # Send current state using standardized message format
                        state_message = create_message(
                            "state_update",
                            payload,
                            tick=current_tick,
                            client_id=client_id
                        )
                        await client["ws"].send_json(state_message)
                    except Exception as e:
                        # Client connection is broken, mark for removal
                        print(f"Error sending to client {client_id}: {e}")
                        to_remove.append(client_id)
                
                # Print when we sent state updates
                if len(clients) > 0:
                    print(f"[TICK {current_tick}] State update sent at {state_send_time:.6f}")
                
                # Remove disconnected clients and clean up their ships
                for client_id in to_remove:
                    client = clients[client_id]
                    if client.get('ship_id'):  # If client had a ship
                        env.remove_ship(client['ship_id'])  # Remove from physics
                        remove_from_leaderboard(client['ship_id'])  # Remove from leaderboard
                        if client['ship_id'] in trail_history:
                            del trail_history[client['ship_id']]  # Remove trail data
                    clients.pop(client_id, None)  # Remove client from tracking
                
                # Check if baseline ship has been kicked off orbit (died)
                if baseline_ship_id in env.ships:
                    ship = env.ships[baseline_ship_id]
                    r = np.sqrt(ship['x']**2 + ship['y']**2)  # Distance from center
                    if r > 10.0:  # Baseline ship went too far out
                        print("Baseline ship kicked off orbit!")
                        # Only reset environment if no clients are connected
                        if not clients:
                            env.reset()  # Reset physics environment
                            trail_history.clear()  # Clear all trails
                            add_baseline_ship()  # Add new baseline ship
                            env.current_step = 0  # Reset step counter
        
        # Wait 16.7ms (60Hz simulation rate) before next physics step
        await asyncio.sleep(0.0167)

######################## START & STOP SIMULATION ########################
async def start_simulation():
    """Start the global physics simulation loop"""
    global simulation_running, simulation_task
    
    if not simulation_running:  # Only start if not already running
        simulation_running = True  # Set flag to start simulation
        # Add baseline ship when simulation starts (first client joins)
        add_baseline_ship()
        # Create async task to run the simulation loop
        simulation_task = asyncio.create_task(global_simulation_loop())
        print("Global simulation loop started with baseline ship")

async def stop_simulation():
    """Stop the global physics simulation loop"""
    global simulation_running, simulation_task
    
    if simulation_running:  # Only stop if currently running
        simulation_running = False  # Set flag to stop simulation
        if simulation_task:
            simulation_task.cancel()  # Cancel the async task
            try:
                await simulation_task  # Wait for task to finish
            except asyncio.CancelledError:
                pass  # Expected when cancelling task
        # Remove baseline ship when simulation stops
        remove_baseline_ship()
        print("Global simulation loop stopped")

######################## WEBSOCKET ENDPOINT ########################
# WebSocket endpoint that handles individual client connections
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()  # Accept the WebSocket connection
    except Exception as e:
        print(f"Error accepting WebSocket connection: {e}")
        return
    
    # Generate unique ID for this client
    client_id = str(uuid.uuid4())
    
    try:
        # Automatically put new client in observe mode
        async with clients_lock:  # Prevent race conditions
            clients[client_id] = {
                "ws": websocket,  
                "type": "observer", 
                "ship_id": None 
            }
            
            # Start simulation if this is the first client to connect
            if not simulation_running:
                await start_simulation()
        
        # Send confirmation to client that they're in observe mode upon connection
        confirmation_message = create_message(
            "mode_confirmed",
            {
                "mode": "observe",  
                "message": "You are now observing the simulation"
            },
            client_id=client_id
        )
        await websocket.send_json(confirmation_message)
        

        # Main message handling loop for this client
        while True:
            try:
                # Wait for message from client
                raw_data = await websocket.receive_json()
                
                # Parse standardized message format
                if "header" in raw_data and "payload" in raw_data:
                    header = raw_data["header"]
                    payload = raw_data["payload"]
                    message_type = header.get("type")
                    message_tick = header.get("tick", 0)
                else:
                    # Fallback for old format (backward compatibility)
                    message_type = raw_data.get("type") or raw_data.get("message_type")
                    payload = raw_data
                    message_tick = 0
                
                # Handle mode change requests (observe -> manual/model)
                if message_type == "join_mode":
                    mode = payload.get("mode")  # Which mode to switch to
                    
                    # Prevent race conditions during mode change
                    async with clients_lock:  
                        if mode == "manual":
                            # Manual control mode: create ship that responds to arrow keys
                            # Generate unique ship ID
                            ship_id = str(uuid.uuid4()) 
                            env.add_ship(ship_id, control_type='manual') 
                            add_to_leaderboard(ship_id, f"Manual {ship_id[:8]}")

                            # Update client metadata
                            clients[client_id] = {
                                "ws": websocket, 
                                "type": "manual",  # Manual control type
                                "ship_id": ship_id,  # This client's ship ID
                                "pending_action": {'turn': 0.0, 'thrust': 0.0}  # Initialize action
                            }

                            # Send confirmation to client
                            confirmation_message = create_message(
                                "mode_confirmed",
                                {
                                    "mode": "manual",
                                    "ship_id": ship_id,  # Tell client their ship ID
                                    "message": "You can now control your ship with arrow keys"
                                },
                                client_id=client_id
                            )
                            await websocket.send_json(confirmation_message)
                            
                        elif mode == "model":
                            # Model control mode: wait for model upload before creating ship
                            print(f"Client {client_id} set to model mode, awaiting model upload")
                            
                            # Update client metadata (no ship or model yet)
                            clients[client_id] = {
                                "ws": websocket, 
                                "type": "model",  # Model control type
                                "ship_id": None,  # No ship until model is uploaded
                                "model": None,  # No model loaded yet - client will upload
                                "model_loaded": False
                            }
                            # Send confirmation to client - request model upload
                            confirmation_message = create_message(
                                "mode_confirmed",
                                {
                                    "mode": "model",
                                    "ship_id": None,  # No ship yet
                                    "message": "Please upload your model .zip file",
                                    "awaiting_model": True
                                },
                                client_id=client_id
                            )
                            await websocket.send_json(confirmation_message)
                
                # Handle model file upload
                elif message_type == "model_upload":
                    client = clients.get(client_id)
                    if client and client['type'] == 'model' and not client.get('model_loaded', False):
                        try:
                            # Get base64 encoded model data
                            model_data = payload.get("model_data")
                            if model_data:
                                # Decode base64 and save temporarily
                                import base64
                                import tempfile
                                import os
                                
                                model_bytes = base64.b64decode(model_data)
                                
                                # Save to temporary file
                                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                                    temp_file.write(model_bytes)
                                    temp_model_path = temp_file.name
                                
                                # Load the model
                                try:
                                    uploaded_model = PPO.load(temp_model_path)
                                    
                                    # Create ship now that model is successfully loaded
                                    ship_id = str(uuid.uuid4()) 
                                    env.add_ship(ship_id) 
                                    add_to_leaderboard(ship_id, f"Model {ship_id[:8]}")
                                    
                                    # Update client with loaded model and ship
                                    client['model'] = uploaded_model
                                    client['model_loaded'] = True
                                    client['ship_id'] = ship_id
                                    
                                    # Clean up temporary file
                                    os.unlink(temp_model_path)
                                    
                                    # Send success confirmation with ship_id
                                    success_message = create_message(
                                        "model_upload_response",
                                        {
                                            "success": True,
                                            "message": "Model uploaded successfully! Your ship is now AI controlled.",
                                            "ship_id": ship_id
                                        },
                                        client_id=client_id
                                    )
                                    await websocket.send_json(success_message)
                                    print(f"Model uploaded successfully for client {client_id}, created ship {ship_id}")
                                    
                                except Exception as model_error:
                                    # Clean up temporary file
                                    if os.path.exists(temp_model_path):
                                        os.unlink(temp_model_path)
                                    
                                    error_message = create_message(
                                        "model_upload_response",
                                        {
                                            "success": False,
                                            "message": f"Failed to load model: {str(model_error)}"
                                        },
                                        client_id=client_id
                                    )
                                    await websocket.send_json(error_message)
                                    print(f"Failed to load model for client {client_id}: {model_error}")
                            else:
                                error_message = create_message(
                                    "model_upload_response",
                                    {
                                        "success": False,
                                        "message": "No model data received"
                                    },
                                    client_id=client_id
                                )
                                await websocket.send_json(error_message)
                                
                        except Exception as e:
                            error_message = create_message(
                                "model_upload_response",
                                {
                                    "success": False,
                                    "message": f"Upload error: {str(e)}"
                                },
                                client_id=client_id
                            )
                            await websocket.send_json(error_message)
                            print(f"Model upload error for client {client_id}: {e}")
                
                # Handle manual control input (arrow key presses)
                elif message_type == "manual_action":
                    client = clients.get(client_id)  
                    if client and client['type'] == 'manual':
                        ship_id = client.get('ship_id')
                        if ship_id:
                            # Validate tick
                            is_valid, tick_status = validate_message_tick(message_tick, ship_id)
                            if is_valid:
                                turn = payload.get("turn", 0.0)  
                                thrust = payload.get("thrust", 0.0)
                                action = {'turn': turn, 'thrust': thrust}
                                
                                # Store action for lockstep simulation
                                pending_actions[client_id] = action
                                
                                # Calculate latency from action request to action received
                                action_received_time = asyncio.get_event_loop().time()
                                latency = (action_received_time - last_action_request_time) * 1000  # Convert to milliseconds
                                
                                print(f"ACK Manual action at tick {message_tick}: Action request sent {last_action_request_time:.6f}, Action received {action_received_time:.6f}, Latency {latency:.2f}ms")
                            else:
                                print(f"NACK Manual action from client {client_id} - tick {message_tick} is {tick_status}")
                
                # Handle cancel control request (return to observe mode)
                elif message_type == "cancel_control":
                    client = clients.get(client_id)  # Get this client's data
                    if client and client.get('ship_id'):  # If client has a ship
                        # Remove the ship from physics environment
                        env.remove_ship(client['ship_id'])
                        remove_from_leaderboard(client['ship_id'])
                        if client['ship_id'] in trail_history:
                            del trail_history[client['ship_id']]
                        
                        # Reset client to observer mode
                        clients[client_id] = {
                            "ws": websocket, 
                            "type": "observer",  # Back to observer mode
                            "ship_id": None  # No ship
                        }
                        
                        # Send confirmation to client
                        confirmation_message = create_message(
                            "mode_confirmed",
                            {
                                "mode": "observe",
                                "message": "You are now observing the simulation"
                            },
                            client_id=client_id
                        )
                        await websocket.send_json(confirmation_message)
                        print(f"Client {client_id} cancelled control and returned to observer mode")
                        
            except asyncio.TimeoutError:
                # No message received within timeout, continue loop
                pass
            except (RuntimeError, ConnectionResetError, WebSocketDisconnect):
                # WebSocket connection is broken, exit the loop
                break
            except Exception as e:
                # Log other errors but continue
                print(f"WebSocket receive error: {e}")
                break
                
    except WebSocketDisconnect:
        # Client disconnected normally
        pass
    finally:
        # Clean up when connection ends (client disconnected or error)
        try:
            async with clients_lock:  # Prevent race conditions during cleanup
                client = clients.get(client_id)  # Get client data
                if client and client.get('ship_id'):  # If client had a ship
                    env.remove_ship(client['ship_id']) 
                    remove_from_leaderboard(client['ship_id'])  # Remove from leaderboard
                    if client['ship_id'] in trail_history:
                        del trail_history[client['ship_id']]  # Remove trail data
                clients.pop(client_id, None)  # Remove client from tracking
                
                # Stop simulation if no clients left (but baseline ship continues)
                if not clients and simulation_running:
                    await stop_simulation()
        except RuntimeError as e:
            # Handle event loop issues gracefully
            print(f"Error during client cleanup for {client_id}: {e}")
            # Still try to clean up without the lock
            try:
                client = clients.get(client_id)
                if client and client.get('ship_id'):
                    env.remove_ship(client['ship_id'])
                    remove_from_leaderboard(client['ship_id'])
                    if client['ship_id'] in trail_history:
                        del trail_history[client['ship_id']]
                clients.pop(client_id, None)
            except Exception as cleanup_error:
                print(f"Failed to cleanup client {client_id}: {cleanup_error}")

# Cleanup function when server shuts down
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up when server shuts down"""
    await stop_simulation()  # Stop the simulation loop


######################## RUN SERVER ########################
if __name__ == "__main__":
    import uvicorn  # ASGI server for running FastAPI
    uvicorn.run(app, host="0.0.0.0", port=5501)  # Run server on all interfaces, port 5500 
    

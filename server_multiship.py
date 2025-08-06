import asyncio
import uuid
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from stable_baselines3 import PPO
from environment import MultiShipOrbitalEnvironment

# Lock to prevent RuntimeError: dictionary changed size during iteration 
clients_lock = asyncio.Lock()

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment and clients
env = MultiShipOrbitalEnvironment()
clients = {}  # ship_id: {"ws": ws, "model": model}
model_path = './models/model_22-51-24-_18-05-2025/ppo_orbital_model.zip'

# Helper from OrbitalEnvrionment: convert raw state to processed observation (exact same as render_multiship_trained.py)
def convert_state(state, env, ship):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    r = max(r, 1e-5)
    v_radial = (x * vx + y * vy) / r
    v_tangential = (x * vy - y * vx) / r
    initial_r = ship['init_r']
    flag = 1.0 if np.abs(r - 1.0) < 0.01 else 0.0
    specific_energy = 0.5 * (vx**2 + vy**2) - env.GM / r
    angular_momentum = r * v_tangential
    r_err = r - 1.0
    r_max_err = max(abs(initial_r - 1), 1e-2)
    scaled_r_err = np.clip((r_err / r_max_err) * 2, -2, 2)
    obs = np.array([
        scaled_r_err,
        v_radial,
        v_tangential,
        1 - initial_r,
        flag,
        specific_energy,
        angular_momentum
    ], dtype=np.float32)
    return obs

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
    except Exception as e:
        print(f"Error accepting WebSocket connection: {e}")
        return
    
    # Assign unique ship ID and add to environment
    ship_id = str(uuid.uuid4())
    env.add_ship(ship_id)
    
    # Load PPO model for this ship
    model = PPO.load(model_path)
    clients[ship_id] = {"ws": websocket, "model": model}
    
    try:
        # Send ship ID to client
        await websocket.send_json({"type": "ship_assigned", "ship_id": ship_id})
        
        # Main simulation loop
        while True:
            # Wait for client message (could be used for manual control later)
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.0167)  # 60Hz timeout
                # Handle client messages if needed
            except asyncio.TimeoutError:
                pass
            except (RuntimeError, ConnectionResetError, WebSocketDisconnect):
                # WebSocket connection is broken, break out of loop
                break
            except Exception as e:
                # Log other errors but continue
                print(f"WebSocket receive error: {e}")
                break
            
            # Step environment if we have clients
            async with clients_lock:
                if clients:
                    actions = {}
                    for sid, client in list(clients.items()):
                        ship = env.ships.get(sid)
                        if ship is None or ship.get('done', False):
                            actions[sid] = 0.0
                            continue
                        state = np.array([ship['x'], ship['y'], ship['vx'], ship['vy']], dtype=np.float32)
                        obs = convert_state(state, env, ship)
                        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                        with torch.no_grad():
                            action, _ = client["model"].predict(obs_tensor, deterministic=True)
                        actions[sid] = float(action[0])
                    env.step(actions)

                    # actions = {}
                    # for sid, client in list(clients.items()):
                    #     if not env.ships[sid]['done']:
                    #         ship = env.ships[sid]
                    #         state = np.array([ship['x'], ship['y'], ship['vx'], ship['vy']], dtype=np.float32)
                    #         obs = convert_state(state, env, ship)
                    #         obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                    #         with torch.no_grad():
                    #             action, _ = client["model"].predict(obs_tensor, deterministic=True)
                    #         actions[sid] = float(action[0])
                    #     else:
                    #         actions[sid] = 0.0
           
                
            # Broadcast all ship states to all clients
            states = env.get_states()

            async with clients_lock:
                    to_remove = []
                    for sid, client in list(clients.items()):
                        try:
                            await client["ws"].send_json({
                                "type": "state_update",
                                "ships": states,
                                "your_ship_id": sid
                            })
                        except Exception as e:
                            print(f"Error sending to client {sid}: {e}")
                            to_remove.append(sid)

                    for sid in to_remove:
                        env.remove_ship(sid)
                        clients.pop(sid, None)


                # for sid, client in clients.items():
                #     try:
                #         await client["ws"].send_json({
                #             "type": "state_update",
                #             "ships": states,
                #             "your_ship_id": sid
                #         })
                #     except (RuntimeError, ConnectionResetError, WebSocketDisconnect):
                #         # Remove disconnected client
                #         env.remove_ship(sid)
                #         if sid in clients:
                #             del clients[sid]
                #     except Exception as e:
                #         print(f"Error sending to client {sid}: {e}")
                #         # Remove problematic client
                #         env.remove_ship(sid)
                #         if sid in clients:
                #             del clients[sid]
            
            await asyncio.sleep(0.0167)  # 60Hz update rate (for much faster animation)
            
    # except WebSocketDisconnect:
    #     # Remove ship and client
    #     env.remove_ship(ship_id)
    #     if ship_id in clients:
    #         del clients[ship_id]
    except WebSocketDisconnect:
        pass
    finally:
    # finally = When this WebSocket connection handler (websocket_endpoint) ends,
    # whether due to a normal exit, an exception, or a WebSocket disconnect,
    # make sure to clean up this ship and remove its client from the global state.
        env.remove_ship(ship_id)
        async with clients_lock:
            clients.pop(ship_id, None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5500) 
    

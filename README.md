# Grav-Nav-RL-Multiplayer
Brown CPFU Vers. 8.28.25
## A Fork of https://github.com/BrownParticleAstro/Grav-Nav-RL.git but includes multiship orbital environment & multiplayer game using authoritative server-client setup.

## How To Run Multiplayer Game Locally

### Prerequisites
- Python 3.8 or higher
- A modern web browser
- VS Code with Live Server extension (recommended)

### Step 1: Set Up Python Environment

1. **Create a virtual environment:**
   ```bash
   python3 -m venv grav-nav-env
   ```

2. **Activate the virtual environment:**
   - **On macOS/Linux:**
     ```bash
     source grav-nav-env/bin/activate
     ```
   - **On Windows:**
     ```bash
     grav-nav-env\Scripts\activate
     ```

### Step 2: Install Dependencies

1. **Install required Python packages:**
   ```bash
   pip install fastapi uvicorn websockets numpy torch stable-baselines3 gym matplotlib
   ```

2. **Verify installation:**
   ```bash
   python -c "import fastapi, uvicorn, numpy, torch, stable_baselines3; print('All dependencies installed successfully!')"
   ```

### Step 3: Run the Multiplayer Server

1. **Navigate to the project directory:**
   ```bash
   cd /path/to/Grav-Nav-RL-Server
   ```

2. **Start the server:**
   ```bash
   python server_multiship.py
   ```

3. **Verify server is running:**
   - You should see output like: `INFO: Uvicorn running on http://0.0.0.0:5500`
   - The server will be accessible at `http://localhost:5500`

### Step 4: Open the Game Client

1. **Open VS Code** in the project directory

2. **Install Live Server extension** (if not already installed):
   - Open Extensions (Ctrl+Shift+X)
   - Search for "Live Server"
   - Install the extension by Ritwick Dey

3. **Open the game client:**
   - Open `index.html` in VS Code
   - Right-click on `index.html` in the file explorer
   - Select "Open with Live Server"

4. **The game should now be running:**
   - Your browser will open automatically to `http://127.0.0.1:5500` (or similar)
   - You should see the orbital simulation interface

### Step 5: Join the Game

1. **Choose your control mode:**
   - Click "Manual Control" to control your ship with keyboard
   - Click "RL Model Control" to let the AI control your ship

2. **Manual Controls (if selected):**
   - **↑** Thrust
   - **←** Steer Left 10 degrees
   - **→** Steer Right 10 degrees
   - **Space** No thrust

3. **Multiple players:**
   - Open additional browser tabs/windows to `http://127.0.0.1:5500`
   - Each tab represents a different player
   - All players will see each other's ships in real-time

### Troubleshooting

**Server won't start:**
- Check if port 5500 is already in use: `lsof -i :5500`
- Kill any processes using the port: `kill -9 <PID>`
- Try a different port by modifying `uvicorn.run(app, host="0.0.0.0", port=5500)` in `server_multiship.py`

**Client can't connect:**
- Ensure the server is running and accessible at `http://localhost:5500`
- Check browser console for WebSocket connection errors
- Verify firewall settings aren't blocking the connection

**Ships not moving properly:**
- Check that the model file exists at `./models/model_22-51-24-_18-05-2025/ppo_orbital_model.zip`
- Verify the server console for any error messages
- Try refreshing the browser page

**Performance issues:**
- Close unnecessary browser tabs
- Reduce the update rate by modifying the `await asyncio.sleep(0.0167)` value in `server_multiship.py`
- Check system resources (CPU/memory usage)

---

## 🔄 Communication Schema

### Server State Management
- **`clients` dict**: `{ship_id: {"ws": websocket, "model": model}}`
- **`env.ships` dict**: `{ship_id: {x, y, vx, vy, init_r, done, control_type, heading, thrust, turn_rate, steps}}`
- **`clients_lock`**: Async lock for adding/removing clients

### Client State Management  
- **`myShipId`**: The client's assigned ship ID
- **`ships` dict**: `{ship_id: THREE.Mesh object}` for rendering
- **`connected`**: WebSocket connection status
- **`currentMode`**: 'observe', 'manual', or 'model'

### Message Types

**Server → Client:**
- `{"type": "ship_assigned", "ship_id": "uuid"}` - Assigns ship ID to new client
- `{"type": "mode_confirmed", "ship_id": "uuid", "mode": "manual|model"}` - Confirms control mode
- `{"type": "state_update", "ships": {...}, "your_ship_id": "uuid"}` - Broadcasts all ship states

**Client → Server:**
- `{"type": "join_mode", "mode": "manual|model"}` - Request to join with specific control mode
- `{"type": "manual_action", "action": float}` - Manual control input (for manual mode)

### Control Flow
1. Client connects → Server assigns ship ID → Client receives `ship_assigned`
2. Client sends `join_mode` → Server confirms with `mode_confirmed`
3. Server runs simulation loop:
   - Processes AI actions for model-controlled ships
   - Processes manual actions for manually-controlled ships
   - Steps environment with all actions
   - Broadcasts `state_update` to all clients
4. Client renders ship positions and handles user input

---

## Orbital Simulation Project 🛰️🚀

Simulate 2D orbital maneuvers using **Reinforcement Learning (PPO)**, featuring dynamic visualizations, and customizable training and testing setups.

**Test your PPO ship model in a multi-ship orbital simulations**, to see if your ship _is the last one standing._


## 🌟 Overview

The goal of this project is to simulate spacecraft orbital control. The simulation hooks a model of the user's choice up to controlling thrusts from a spacecraft with the goal of achieving stable orbits under minimal fuel cost

**Core features include:**
- **Training**: Learn optimal control strategies with PPO.
- **Testing**: Evaluate model performance.
- **Rendering**: Visualize the simulation at various stages.
- **Rendering Multiship**: To visualize a simulation of multiple ships orbiting around the same center mass, run the MultipleOrbitalEnvrionment locally by running ```python3 render_multiship_trained.py``` in the terminal. Note that these ships rendered in the multiship environment though are only trained on single body, single ship simulations. They still perform pretty well though.
---

## 📂 Project Structure

### 🛠️ Files & Their Functions

| File Name              | Description                                                                                                                                              |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`environment.py`**   | Defines `OrbitalEnv` (core functionality) & `OrbitalEnvWrapper` (constructing features out of that x,y core) classes. Adjust simulation physics (e.g., thrust, gravity) and modify the reward functions for custom behaviors. |
| **`model.py`**         | Contains the neural network architecture. Customize feature extractors and experiment with different configurations for better learning efficiency.       |
| **`render.py`**        | Handles visualization with `matplotlib`. Register custom figure generation functions for personalized data analysis.                                      |
| **`run_and_view_episode.py`** | Primary script for running, testing, and visualizing full episodes. Organizes the full pipeline, from loading models to rendering results.                  |
| **`train.py`**         | Trains the PPO model. Modify hyperparameters (learning rate, gamma, etc.)       |
| **`test.py`**          | Runs multiple test episodes to evaluate performance. Adjust test settings and load different model weights for comparisons.                                |

---

## 🎯 Customization Options

### ⚙️ Adapt the Project By:
- **Modifying Environment Dynamics**: Adjust thrust, gravity, or the reward structure for varied orbital challenges.
- **Customizing Neural Network Architecture**: Tweak the `create_model` function for different PPO policy structures.
- **Optimizing Training Setup**: Change PPO hyperparameters for better training outcomes.
- **Expanding Visualization**: Create new visual plots for deeper performance insights.

---

## 🖼️ Visualizations & Rendering

**Default Visualization (`"combined"`) Includes:**
- **Radius Over Time** 📈: Line plot showing radial distance over timesteps.
- **Action Over Time** 🔧: Line plot depicting thrust actions taken at each step.
- **Orbit Plot** 🌌: 2D plot showing the spacecraft’s position with velocity and thrust vectors.

### 🛠️ How to Add Custom Visualizations:
1. **Create** a figure generation function.
2. **Register** it in the `Renderer` class.
3. **Call** the function when rendering.

---

## 🚀 Example Usage

### 🔄 Training
Run `train.py`:
```bash
python train.py
```
*Modifies model configurations, saves trained data to a specified path.*

### 🧪 Testing
Run `test.py`:
```bash
python test.py --model_path /path/to/model
```
*Load trained models, run tests, and save results for analysis.*

### 🎬 Running and Visualizing Full Episodes
Run `run_and_view_episode.py` as the primary script to execute, test, and visualize episodes:
```bash
python run_and_view_episode.py --model_path /path/to/model --episode_id 1
```
*This script loads models, runs episodes, and generates visual outputs.*

### 📊 Rendering
Use `render.py`:
```bash
python render.py --model_path /path/to/model --episode_id 1
```
*Visualize selected episodes, add custom plots as needed.*

---

Enjoy experimenting with the orbital environment, and feel free to adapt this project to meet your research or educational needs! 🌍✨

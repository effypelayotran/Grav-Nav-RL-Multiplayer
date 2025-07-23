import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
from stable_baselines3 import PPO
from environment import MultiShipOrbitalEnvironment
import torch

# Parameters
num_ships = 3
num_steps = 5000
model_path = './models/model_22-51-24-_18-05-2025/ppo_orbital_model.zip'  # Update if needed

# Load trained PPO model
model = PPO.load(model_path)

# Helper: convert raw state to processed observation (as in OrbitalEnvWrapper._convert_state)
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

# Initialize environment and ships
env = MultiShipOrbitalEnvironment()
ship_ids = []
for i in range(num_ships):
    ship_id = f'ship_{i}'
    env.add_ship(ship_id)
    ship_ids.append(ship_id)

# Record trajectories and actions
trajectories = {sid: {'x': [], 'y': [], 'vx': [], 'vy': [], 'action': []} for sid in ship_ids}

# Simulate
for t in range(num_steps):
    actions = {}
    for sid in ship_ids:
        ship = env.ships[sid]
        state = np.array([ship['x'], ship['y'], ship['vx'], ship['vy']], dtype=np.float32)
        obs = convert_state(state, env, ship)
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            action, _ = model.predict(obs_tensor, deterministic=True)
        actions[sid] = float(action[0])
    env.step(actions)
    for sid in ship_ids:
        ship = env.ships[sid]
        trajectories[sid]['x'].append(ship['x'])
        trajectories[sid]['y'].append(ship['y'])
        trajectories[sid]['vx'].append(ship['vx'])
        trajectories[sid]['vy'].append(ship['vy'])
        trajectories[sid]['action'].append(actions[sid])

# Animation setup
colors = ['red', 'blue', 'green', 'yellow', 'purple']
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.set_title('Multi-Ship PPO Model Animation')
ax.plot(0, 0, marker='*', markersize=20, color='orange', label='Central Mass')

# Draw reference circles
for radius in range(1, 6):
    circle = Circle((0, 0), radius, color=(0.5, 0.5, 0.5, 0.3), fill=False, linestyle='--')
    ax.add_artist(circle)

lines = []
points = []
vel_arrows = []
thrust_arrows = []
for i, sid in enumerate(ship_ids):
    (line,) = ax.plot([], [], color=colors[i], lw=2, label=f'Ship {i} Trajectory')
    (point,) = ax.plot([], [], marker='o', color=colors[i], markersize=12, label=f'Ship {i} Position')
    lines.append(line)
    points.append(point)
    vel_arrows.append(None)
    thrust_arrows.append(None)
ax.legend()

# Animation update function
def update(frame):
    for i, sid in enumerate(ship_ids):
        # Trajectory
        lines[i].set_data(trajectories[sid]['x'][:frame], trajectories[sid]['y'][:frame])
        # Current position
        points[i].set_data([trajectories[sid]['x'][frame-1]], [trajectories[sid]['y'][frame-1]])
        # Remove previous arrows
        if vel_arrows[i]:
            vel_arrows[i].remove()
        if thrust_arrows[i]:
            thrust_arrows[i].remove()
        # Velocity arrow
        x = trajectories[sid]['x'][frame-1]
        y = trajectories[sid]['y'][frame-1]
        vx = trajectories[sid]['vx'][frame-1]
        vy = trajectories[sid]['vy'][frame-1]
        vel_arrows[i] = ax.arrow(x, y, vx*0.2, vy*0.2, head_width=0.15, head_length=0.25, fc=colors[i], ec=colors[i], alpha=0.7)
        # Thrust arrow (tangential)
        action = trajectories[sid]['action'][frame-1]
        r = np.sqrt(x**2 + y**2)
        if r > 1e-5:
            rhat = np.array([x, y]) / r
            tang = np.array([-rhat[1], rhat[0]])
            tx, ty = tang * action * 0.5
            thrust_arrows[i] = ax.arrow(x, y, tx, ty, head_width=0.12, head_length=0.18, fc='orange', ec='orange', alpha=0.7)
    return lines + points + [a for a in vel_arrows if a] + [a for a in thrust_arrows if a]



frame_skip = 1
ani = animation.FuncAnimation(fig, update, frames=range(0, num_steps, frame_skip), interval=20, blit=True)

plt.show() 
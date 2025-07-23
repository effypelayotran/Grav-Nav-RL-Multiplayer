# Grav-Nav-RL-Server
Brown CPFU Vers. 7.23.25

## Orbital Simulation Project 🛰️🚀

Simulate 2D orbital maneuvers using **Reinforcement Learning (PPO)**, featuring dynamic visualizations, and customizable training and testing setups.

Test your PPO ship model in a multi-ship orbital simulations, to see if your ship is the last one standing.

---

## 🌟 Overview

The goal of this project is to simulate spacecraft orbital control. The simulation hooks a model of the user's choice up to controlling thrusts from a spacecraft with the goal of achieving stable orbits under minimal fuel cost

**Core features include:**
- **Training**: Learn optimal control strategies with PPO.
- **Testing**: Evaluate model performance.
- **Rendering**: Visualize the simulation at various stages.
- **Rendering Multiship**: Rendering MutipleOrbitalEnvironment both locally and on a live server via server_multiship.py. Guide for setting up and developing the live server is here:  https://docs.google.com/document/d/1xjSoBIB6DazPc7yJHS8tudBpW7BRP2kl27tRpOFXGAk/edit?usp=sharing 

To run the MultipleOrbitalEnvrionment locally, run python3 render_multiship_trained.py. Note that these ships rendered in the multiship environment though are only trained on single body, single ship simulations. They still perform pretty well though.

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

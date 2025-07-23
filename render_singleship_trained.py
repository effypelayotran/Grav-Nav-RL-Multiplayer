import os
from environment import OrbitalEnvWrapper
from stable_baselines3 import PPO
from render import Renderer
from model import create_model
from train import train_model
from test import test_model

# Create the training environment
env_train = OrbitalEnvWrapper()

# Train the model
save_dir = './models'
model_save_path = './models/model_22-51-24-_18-05-2025'

# Load the trained model for inference and testing
env_test = OrbitalEnvWrapper()
for i in range(1):
    env_test.reset()
    test_model(env_test, os.path.join(model_save_path, "ppo_orbital_model"), model_save_path, episode_num=1)

    # Create a renderer instance using the dynamic model_save_path
    renderer = Renderer(model_save_path=model_save_path)

    # Render the first episode from the training data
    renderer.render(episode_num=1, interval=50, data_type="testing")
    env_test.reset()


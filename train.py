import os
import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from model import create_model

# SaveBest is a custom callback that saves the best model during training based on average episode reward.
class SaveBest(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(SaveBest, self).__init__(verbose)
        self.save_path = os.path.join(save_path, 'training')  # Create a "training" folder
        os.makedirs(self.save_path, exist_ok=True)  # Ensure the folder exists
        self.episode_data = []
        self.best_mean_reward = -float('inf')
        self.episode_num = 0  # Track episode number for saving
        self.episode_step = 0  # Track number of steps in the current episode

    def _on_step(self) -> bool:
        """
        This method is called at each step in the environment.
        """
        info = self.locals["infos"][0]
        state = info["state"]
        action = np.squeeze(self.locals["actions"])
        r_err_norm = info["r_err_norm"]
        d_r_err_norm = info["d_r_err_norm"]
        int_r_err_norm = info["int_r_err_norm"]

        # Append data with new quantities
        self.episode_data.append((
            *state, self.episode_step, action, r_err_norm, d_r_err_norm, int_r_err_norm
        ))

        self.episode_step += 1

        # Check if the episode has finished
        done = self.locals["dones"][0]
        if done:
            self.save_episode_data(self.episode_data, self.episode_num)
            self.episode_num += 1
            self.episode_data = []
            self.episode_step = 0

        return True

    def save_episode_data(self, episode_data, episode_num):
        """
        Saves the current episode data to a file.
        """
        # Save the current episode data to a .npz file, ensuring each array is properly dimensioned
        np.savez(os.path.join(self.save_path, f'episode_{episode_num}.npz'),
                x=np.array([step[0] for step in episode_data]),
                y=np.array([step[1] for step in episode_data]),
                vx=np.array([step[2] for step in episode_data]),
                vy=np.array([step[3] for step in episode_data]),
                episode_step=np.array([step[4] for step in episode_data]),
                action=np.array([step[5] for step in episode_data]),
                r_err_norm=np.array([step[6] for step in episode_data]),
                d_r_err_norm=np.array([step[7] for step in episode_data]),
                int_r_err_norm=np.array([step[8] for step in episode_data]))

"""
    Train the model in "env" environment for X number of timesteps with Y reward threshold to stop training
    Save into particular directory
"""
def train_model(env, save_dir, total_timesteps=10_000):
    """
    Args:
        env: Training environment (gym.Env).
        save_dir: Directory to save the model and training data (str).
        total_timesteps: Number of timesteps to train (int).

    Returns:
        Tuple of (trained model, model save path).
    """
    # Create a unique folder for this specific model using timestamp
    model_name = f"model_{datetime.datetime.now().strftime('%H-%M-%S-_%d-%m-%Y')}"
    model_save_path = os.path.join(save_dir, model_name)
    os.makedirs(model_save_path, exist_ok=True)

    callback = SaveBest(save_path=model_save_path)
    model = create_model(env)
    
    # Train the model and save weights
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(os.path.join(model_save_path, "ppo_orbital_model"))

    print(f"Training completed. Model and data saved in {model_save_path}")
    return model, model_save_path

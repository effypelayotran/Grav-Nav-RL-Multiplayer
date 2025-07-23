import gym
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from itertools import chain

# Custom NewtonOptimizer class
class NewtonOptimizer(optim.Optimizer):
    """
    Implements a simplified version of Newton's Method for deep learning.
    Computes parameter updates using the inverse of the Hessian matrix.
    """
    def __init__(self, params, lr=1.0, damping=1e-4):
        """
        Args:
            params: Iterable of model parameters to optimize.
            lr: Learning rate for the parameter updates.
            damping: Damping factor for stabilizing the Hessian inversion.
        """
        defaults = {'lr': lr, 'damping': damping}
        super(NewtonOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Args:
            closure: A closure that re-evaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('NewtonOptimizer does not support sparse gradients.')

                hessian = self._compute_hessian(grad, p)
                hessian_inv = torch.linalg.pinv(hessian + group['damping'] * torch.eye(hessian.size(0)))

                update = hessian_inv @ grad.view(-1)
                p.data.add_(-group['lr'] * update.view(p.size()))

        return loss

    def _compute_hessian(self, grad, param):
        """
        Computes the Hessian matrix for the given gradient.
        """
        grad2rd = torch.autograd.grad(grad.sum(), param, create_graph=True)[0]
        hessian = []
        for g2 in grad2rd.view(-1):
            h_row = torch.autograd.grad(g2, param, retain_graph=True)[0].view(-1)
            hessian.append(h_row)
        return torch.stack(hessian)

# Custom policy class that uses NewtonOptimizer as the optimizer
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)
    def _make_optimizers(self):
        self.optimizer = NewtonOptimizer(self.parameters(), lr=self.learning_rate, damping=1e-4)

def create_model(env):
    """
    Initializes an untrained PPO model with the default feature extractor.
    """
    policy_kwargs = dict(
        net_arch=[dict(pi=[32, 32, 32, 32, 32, 32],    # Actor network (abritrary deep architecture)
                       vf=[32, 32, 32, 32, 32, 32])],  # Critic network (abritrary deep architecture)
        activation_fn=nn.ReLU,
    )

    return PPO(CustomActorCriticPolicy, # Architecture type w Newton optimizer
        env,                            # Environment
        policy_kwargs=policy_kwargs,
        verbose=1,                   
        learning_rate=3e-4,             # Learning rate
        n_steps=2048,                   # num of steps before policy update
        batch_size=64,                  # num of samples per policy update calculation
        ent_coef=0.01,                  # factor to encourage randomness of policy (aka exploration)
        gamma=0.9995,                   # far-sighted consideration of long term reward
        )              

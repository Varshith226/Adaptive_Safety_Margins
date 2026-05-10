import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class TaskPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        """
        The primary policy trying to solve the task (maximize reward).
        """
        super(TaskPolicy, self).__init__()
        
        self.max_action = max_action
        
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # Clamp log_std to prevent mathematical errors (NaNs) during training
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std

    def select_action(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        dist = Normal(mean, std)
        
        if deterministic:
            action = mean
        else:
            action = dist.rsample()
            
        # Scale the action to fit the environment's physical limits
        action = torch.tanh(action) * self.max_action
        return action
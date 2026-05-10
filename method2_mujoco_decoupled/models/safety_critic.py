import torch
import torch.nn as nn
import torch.nn.functional as F

class SafetyCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        The Safety Critic Q_psi(s, a)
        Outputs the predicted discounted probability of future constraint violations.
        Since it predicts a probability/cost, we will eventually bound the output between 0 and 1.
        """
        super(SafetyCritic, self).__init__()
        
        # Q1 architecture (We use two Q-networks to mitigate overestimation bias, 
        # a standard trick from Soft Actor-Critic, but applied to costs)
        self.q1_layer1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_layer3 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.q2_layer1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_layer3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # Concatenate state and action
        sa = torch.cat([state, action], dim=1)
        
        # Forward pass Q1
        q1 = F.relu(self.q1_layer1(sa))
        q1 = F.relu(self.q1_layer2(q1))
        # We use sigmoid on the final output to ensure the predicted cost is between 0 and 1
        q1 = torch.sigmoid(self.q1_layer3(q1))
        
        # Forward pass Q2
        q2 = F.relu(self.q2_layer1(sa))
        q2 = F.relu(self.q2_layer2(q2))
        q2 = torch.sigmoid(self.q2_layer3(q2))
        
        return q1, q2

    def get_risk(self, state, action):
        """
        When deciding if an action is safe, we conservatively take the 
        MAXIMUM predicted risk from our two critics.
        """
        q1, q2 = self.forward(state, action)
        return torch.max(q1, q2)
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import safety_gymnasium
import gymnasium as gym

# Import your custom modules
from models.task_policy import TaskPolicy
from models.recovery_policy import RecoveryPolicy
from models.safety_critic import SafetyCritic
from utils.asm_scheduler import AdaptiveSafetyMargin
from utils.replay_buffer import ReplayBuffer

def train():
    print("Initializing Environment...")
    env = safety_gymnasium.make('SafetyPointGoal1-v0')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print("Initializing Neural Networks & Memory...")
    task_policy = TaskPolicy(state_dim, action_dim, max_action)
    recovery_policy = RecoveryPolicy(state_dim, action_dim, max_action)
    safety_critic = SafetyCritic(state_dim, action_dim)
    
    # NEW: Initialize the Memory Buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    # NEW: Initialize PyTorch Optimizers (The "learning engines")
    # Learning rate (lr) determines how fast the networks update their weights
    task_optimizer = optim.Adam(task_policy.parameters(), lr=3e-4)
    recovery_optimizer = optim.Adam(recovery_policy.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(safety_critic.parameters(), lr=3e-4)

    # Initialize your novel ASM Scheduler
    asm = AdaptiveSafetyMargin(eps_min=0.1, eps_max=0.8, kappa=10, tau=0.5, burn_in_episodes=10)

    total_episodes = 200
    batch_size = 256 # How many memories to look at during a learning step
    gamma = 0.99     # Discount factor for future predictions
    
    print("\nStarting Training Loop with ASM + Backpropagation...")
    print("-" * 60)
    
    # NEW: Dictionary to store metrics for our graphs
    training_logs = {
        "episode": [],
        "reward": [],
        "cost": [],
        "epsilon": [],
        "rho": []
    }

    for episode in range(1, total_episodes + 1):
        obs, info = env.reset()
        state = obs
        
        episode_reward = 0
        episode_cost = 0
        terminated, truncated = False, False
        
        current_epsilon = asm.get_threshold(episode)

        while not (terminated or truncated):
            # Convert state to tensor for the networks
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 1. Propose action
            proposed_action = task_policy.select_action(state_tensor)
            
            # 2. Safety Critic evaluates danger
            with torch.no_grad():
                predicted_risk = safety_critic.get_risk(state_tensor, proposed_action).item()
            
            # 3. ASM Logic: Intervene or Execute
            if predicted_risk > current_epsilon:
                action = recovery_policy.select_action(state_tensor).detach().numpy()[0]
            else:
                action = proposed_action.detach().numpy()[0]
                
            # 4. Step Environment
            next_obs, reward, cost, terminated, truncated, info = env.step(action)
            done = 1.0 if terminated else 0.0
            
            # 5. NEW: Save this memory to the Replay Buffer!
            replay_buffer.add(state, action, reward, cost, next_obs, done)
            
            episode_reward += reward
            episode_cost += cost
            state = next_obs
            
            # --- NEW: THE LEARNING STEP ---
            # Once we have enough memories saved up, start updating the networks
            if replay_buffer.size > batch_size:
                # Pull a random batch of memories
                batch_states, batch_actions, batch_rewards, batch_costs, batch_next_states, batch_dones = replay_buffer.sample(batch_size)
                
                # A. Update the Safety Critic (Learn to predict costs better)
                # We want the critic to predict: Cost + (Discount * Next Future Cost)
                with torch.no_grad():
                    next_actions = task_policy.select_action(batch_next_states, deterministic=True)
                    target_risk = batch_costs + gamma * (1 - batch_dones) * safety_critic.get_risk(batch_next_states, next_actions)
                
                current_q1, current_q2 = safety_critic(batch_states, batch_actions)
                critic_loss = F.mse_loss(current_q1, target_risk) + F.mse_loss(current_q2, target_risk)
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # B. Update Task Policy (Learn to maximize rewards)
                # (Simplified Policy Gradient step for the PoC)
                task_actions, _ = task_policy(batch_states)
                task_loss = -task_actions.mean() * batch_rewards.mean() # Encourage actions that led to high reward
                
                task_optimizer.zero_grad()
                task_loss.backward()
                task_optimizer.step()

                # C. Update Recovery Policy (Learn to minimize costs)
                recovery_actions, _ = recovery_policy(batch_states)
                recovery_loss = recovery_actions.mean() * batch_costs.mean() # Discourage actions that led to high cost
                
                recovery_optimizer.zero_grad()
                recovery_loss.backward()
                recovery_optimizer.step()
            # -------------------------------

        success = 1.0 if (episode_reward > 0 and episode_cost == 0) else 0.0
        current_rho = asm.update_competence(success)
        
        print(f"Episode {episode:03d} | Epsilon: {current_epsilon:.3f} | Comp (rho): {current_rho:.3f} | Reward: {episode_reward:.1f} | Cost: {episode_cost:.1f}")
    # NEW: Log the data for this episode
        training_logs["episode"].append(episode)
        training_logs["reward"].append(episode_reward)
        training_logs["cost"].append(episode_cost)
        training_logs["epsilon"].append(current_epsilon)
        training_logs["rho"].append(current_rho)
        
    # --- THIS GOES OUTSIDE THE LOOP, AT THE VERY END OF train() ---
    # Save everything to a CSV file when training finishes
    df = pd.DataFrame(training_logs)
    df.to_csv("asm_training_results.csv", index=False)
    print("Training logs saved to 'asm_training_results.csv'")
    
    env.close()
    print("-" * 60)
    print("Training Complete! The networks have updated their weights.")

if __name__ == "__main__":
    train()
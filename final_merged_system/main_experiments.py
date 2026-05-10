import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import collections
import random
import math
import matplotlib
matplotlib.use('Agg') # Headless-safe for server environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import safety_gymnasium


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─── 1. INFRASTRUCTURE & NETWORKS ─────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, cost):
        self.buffer.append((state, action, reward, next_state, done, cost))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d, c = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(DEVICE),
            torch.FloatTensor(np.array(a)).to(DEVICE),
            torch.FloatTensor(np.array(r)).to(DEVICE),
            torch.FloatTensor(np.array(ns)).to(DEVICE),
            torch.FloatTensor(np.array(d)).to(DEVICE),
            torch.FloatTensor(np.array(c)).to(DEVICE),
        )
    def __len__(self) -> int: return len(self.buffer)

class SafetyCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),                  nn.ReLU(),
            nn.Linear(hidden, 1),                       nn.Sigmoid(),
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class TaskCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        def _mlp(sd, ad, h):
            return nn.Sequential(
                nn.Linear(sd + ad, h), nn.ReLU(),
                nn.Linear(h, h),       nn.ReLU(),
                nn.Linear(h, 1),
            )
        self.q1 = _mlp(state_dim, action_dim, hidden)
        self.q2 = _mlp(state_dim, action_dim, hidden)
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

class TaskPolicy(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -20, 2
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.mu      = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, state):
        x = self.fc(state)
        mu  = self.mu(x)
        std = torch.exp(torch.clamp(self.log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX))
        dist = Normal(mu, std)
        z    = dist.rsample()
        a    = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        return a, log_prob.sum(-1, keepdim=True)

# ─── 2. ASM COMPONENTS ────────────────────────────────────────────────────────

class ASMScheduler:
    def __init__(self, e_min=0.05, e_max=0.50, kappa=10.0, tau=0.60, burn_in=50, schedule="sigmoid", total_episodes=250):
        self.e_min, self.e_max = e_min, e_max
        self.kappa, self.tau = kappa, tau
        self.burn_in = burn_in
        self.schedule = schedule
        self.total_episodes = total_episodes

    def get_threshold(self, episode: int, rho: float) -> float:
        if episode <= self.burn_in: return self.e_min
        if self.schedule == "sigmoid":
            sig = 1.0 / (1.0 + math.exp(-self.kappa * (rho - self.tau)))
            return self.e_min + (self.e_max - self.e_min) * sig
        elif self.schedule == "linear":
            progress = (episode - self.burn_in) / max(1, self.total_episodes - self.burn_in)
            return self.e_min + (self.e_max - self.e_min) * min(progress, 1.0)
        elif self.schedule == "step":
            if   rho < 0.2: return self.e_min
            elif rho < 0.4: return self.e_min + (self.e_max - self.e_min) * 0.25
            elif rho < 0.6: return self.e_min + (self.e_max - self.e_min) * 0.50
            elif rho < 0.8: return self.e_min + (self.e_max - self.e_min) * 0.75
            else:           return self.e_max

class CompetenceTracker:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.rho   = 0.0
    def update(self, success: bool) -> float:
        self.rho = self.alpha * float(success) + (1 - self.alpha) * self.rho
        return self.rho

# ─── 3. SAC + ASM AGENT ───────────────────────────────────────────────────────

class ASM_Agent:
    BATCH_SIZE, GAMMA, TAU_SOFT, ALPHA_ENTROPY, RECOVERY_STEPS = 64, 0.99, 0.005, 0.2, 4

    def __init__(self, state_dim: int, action_dim: int):
        self.policy         = TaskPolicy(state_dim, action_dim).to(DEVICE)
        self.critic         = TaskCritic(state_dim, action_dim).to(DEVICE)
        self.critic_target  = TaskCritic(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.safety_critic  = SafetyCritic(state_dim, action_dim).to(DEVICE)

        self.p_opt = optim.Adam(self.policy.parameters(),        lr=1e-3)
        self.c_opt = optim.Adam(self.critic.parameters(),        lr=1e-3)
        self.s_opt = optim.Adam(self.safety_critic.parameters(), lr=1e-3)

    def get_recovery_action(self, state_t, action_task):
        rec = action_task.clone().detach().requires_grad_(True)
        opt = optim.SGD([rec], lr=0.1)
        for _ in range(self.RECOVERY_STEPS):
            risk = self.safety_critic(state_t, torch.tanh(rec))
            opt.zero_grad(); risk.backward(); opt.step()
        return torch.tanh(rec).detach()

    def update(self, buffer: ReplayBuffer):
        if len(buffer) < self.BATCH_SIZE: return
        s, a, r, ns, d, c = buffer.sample(self.BATCH_SIZE)

        # 1. Safety Critic Update
        risk_pred = self.safety_critic(s, a)
        s_loss = F.binary_cross_entropy(risk_pred, c.unsqueeze(1).clamp(0.0, 1.0))
        self.s_opt.zero_grad(); s_loss.backward(); self.s_opt.step()

        # 2. Task Critic Update
        with torch.no_grad():
            na, nlp    = self.policy(ns)
            q1t, q2t   = self.critic_target(ns, na)
            target_q   = r.unsqueeze(1) + (1 - d.unsqueeze(1)) * self.GAMMA * (torch.min(q1t, q2t) - self.ALPHA_ENTROPY * nlp)
        cq1, cq2 = self.critic(s, a)
        c_loss = F.mse_loss(cq1, target_q) + F.mse_loss(cq2, target_q)
        self.c_opt.zero_grad(); c_loss.backward(); self.c_opt.step()

        # 3. Policy Update
        na2, lp2   = self.policy(s)
        q1p, q2p   = self.critic(s, na2)
        p_loss     = (self.ALPHA_ENTROPY * lp2 - torch.min(q1p, q2p)).mean()
        self.p_opt.zero_grad(); p_loss.backward(); self.p_opt.step()

        # 4. Soft Target Update
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.TAU_SOFT * p.data + (1 - self.TAU_SOFT) * tp.data)

# ─── 4. TRAINING LOOP ─────────────────────────────────────────────────────────

@dataclass
class RunResult:
    name: str
    ep_rewards: list = field(default_factory=list)
    ep_costs: list = field(default_factory=list)
    cumulative_costs: list = field(default_factory=list)
    ep_thresholds: list = field(default_factory=list)
    competence: list = field(default_factory=list)
    steps_to_first_success: Optional[int] = None

def train(episodes=250, schedule="sigmoid", fixed_epsilon=0.05, burn_in=50, name="ASM"):
    env = safety_gymnasium.make('SafetyPointGoal1-v0', render_mode=None)
    sd, ad = env.observation_space.shape[0], env.action_space.shape[0]
    agent = ASM_Agent(sd, ad)
    buffer = ReplayBuffer(100_000)
    tracker = CompetenceTracker()
    result = RunResult(name=name)
    scheduler = ASMScheduler(burn_in=burn_in, schedule=schedule, total_episodes=episodes) if schedule != "fixed" else None

    cumulative_cost, total_steps, first_success_ep = 0.0, 0, None

    # Create a directory for this specific run
    checkpoint_dir = f"checkpoints/{name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        ep_reward, ep_cost = 0.0, 0.0
        reached_goal, violated = False, False
        eps = fixed_epsilon if scheduler is None else scheduler.get_threshold(ep, tracker.rho)

        while True:
            st = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad(): at_task, _ = agent.policy(st)

            # ASM Switching Logic
            if agent.safety_critic(st, at_task).item() > eps:
                action = agent.get_recovery_action(st, at_task).squeeze().cpu().numpy()
            else:
                action = at_task.squeeze().cpu().numpy()

            # Safety Gymnasium step unpacks 6 values
            ns, r, cost, term, trunc, info = env.step(action)
            done = float(term or trunc)
            
            buffer.push(state, action, r, ns, done, cost)
            state = ns
            ep_reward += r
            ep_cost += cost
            total_steps += 1

            if cost > 0: violated = True
            if term and not violated: reached_goal = True
            
            if len(buffer) >= agent.BATCH_SIZE and total_steps % 3 == 0:
                agent.update(buffer)
                    
            if term or trunc: break

        # Parameter-free success: The agent stayed safe AND made net-positive progress
        success = (ep_cost == 0) and (ep_reward > 0.0)
        tracker.update(success)
        if success and first_success_ep is None: result.steps_to_first_success = total_steps

        cumulative_cost += ep_cost
        result.ep_rewards.append(ep_reward); result.ep_costs.append(ep_cost)
        result.cumulative_costs.append(cumulative_cost); result.ep_thresholds.append(eps)
        result.competence.append(tracker.rho)

        if ep % 50 == 0:
            print(f"[{name}] Ep {ep:3d}/{episodes} | ε={eps:.3f} | ρ={tracker.rho:.3f} | Ret={ep_reward:+.1f} | Cost={ep_cost:.0f}")
            
            # Save the policy and critics
            torch.save({
                'episode': ep,
                'policy_state_dict': agent.policy.state_dict(),
                'safety_critic_state_dict': agent.safety_critic.state_dict(),
                'optimizer_state_dict': agent.p_opt.state_dict(),
                'competence_rho': tracker.rho
            }, f"{checkpoint_dir}/model_ep{ep}.pt")

    return result

# ─── 5. PLOTTING UTILS ────────────────────────────────────────────────────────

def smooth(x, window=20):
    if len(x) < window: return x
    return np.convolve(x, np.ones(window)/window, mode="same")

def plot_main_comparison(results, save="main_comparison.png"):
    colours = {"Conservative": "#e74c3c", "ASM-Sigmoid": "#2ecc71", "Aggressive": "#3498db"}
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    for r in results: ax1.plot(smooth(r.ep_rewards), label=r.name, color=colours.get(r.name), lw=2)
    ax1.set_title("Episode Return (smoothed)"); ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    for r in results: ax2.plot(r.cumulative_costs, label=r.name, color=colours.get(r.name), lw=2)
    ax2.set_title("Cumulative Safety Cost"); ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    asm_r = next(r for r in results if r.name == "ASM-Sigmoid")
    ax3.plot(asm_r.ep_thresholds, color="#2ecc71", lw=2, label="ε_t")
    ax3.plot(asm_r.competence, color="#f39c12", lw=2, linestyle="--", label="ρ_t (competence)")
    ax3.axvline(50, color="grey", linestyle=":", label="Burn-in end")
    ax3.set_title("ASM: Threshold & Competence"); ax3.legend()

    plt.savefig(save, dpi=150, bbox_inches="tight")
    print(f"Saved -> {save}")

def plot_ablation(results, save="ablation.png"):
    colours = {"ASM-Sigmoid": "#2ecc71", "ASM-Linear": "#f39c12", "ASM-Step": "#9b59b6"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    for r in results: axes[0].plot(smooth(r.ep_rewards), label=r.name, color=colours.get(r.name), lw=2)
    axes[0].set_title("Episode Return (smoothed)"); axes[0].legend()
    
    for r in results: axes[1].plot(r.cumulative_costs, label=r.name, color=colours.get(r.name), lw=2)
    axes[1].set_title("Cumulative Cost"); axes[1].legend()
    
    for r in results: axes[2].plot(r.ep_thresholds, label=r.name, color=colours.get(r.name), lw=2)
    axes[2].axvline(50, color="grey", linestyle=":")
    axes[2].set_title("Threshold ε over Episodes"); axes[2].legend()

    plt.savefig(save, dpi=150, bbox_inches="tight")
    print(f"Saved -> {save}")

# ─── 6. EXECUTION ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    eps = 250
    print("--- Phase 1: Main Comparison ---")
    res_cons = train(eps, schedule="fixed", fixed_epsilon=0.05, name="Conservative")
    res_asm  = train(eps, schedule="sigmoid", name="ASM-Sigmoid")
    res_agg  = train(eps, schedule="fixed", fixed_epsilon=0.50, name="Aggressive")
    plot_main_comparison([res_cons, res_asm, res_agg])

    print("\n--- Phase 2: Ablation Study ---")
    res_lin = train(eps, schedule="linear", name="ASM-Linear")
    res_step = train(eps, schedule="step", name="ASM-Step")
    plot_ablation([res_asm, res_lin, res_step])

   # ─── FINAL EXPORT BLOCK (Add this to the very bottom) ───────────────

    all_results = [res_cons, res_asm, res_agg, res_lin, res_step]
    
    # 1. Episode-by-Episode Data (For line graphs and pareto curves)
    data = []
    for r in all_results:
        for ep in range(len(r.ep_rewards)):
            data.append({
                "Method": r.name,
                "Episode": ep + 1,
                "Return": round(r.ep_rewards[ep], 2),
                "Cost": r.ep_costs[ep],
                "Cumulative Cost": r.cumulative_costs[ep],
                "Epsilon": round(r.ep_thresholds[ep] if r.ep_thresholds else 0, 3),
                "Competence_Rho": round(r.competence[ep] if r.competence else 0, 3)
            })
    pd.DataFrame(data).to_csv("asm_episode_data.csv", index=False)
    
    # 2. Steps to First Success (For your Exploration Efficiency table)
    summary_data = []
    for r in all_results:
        summary_data.append({
            "Method": r.name,
            "Steps to First Success": r.steps_to_first_success if r.steps_to_first_success else "Never Succeeded"
        })
    pd.DataFrame(summary_data).to_csv("asm_exploration_summary.csv", index=False)
    
    # 3. Save the Trained Model Weights
    # This ensures if you ever want to render a video of the agent dodging hazards,
    # you just load this file instead of re-training for hours.
    torch.save(res_asm.policy.state_dict(), "asm_policy_final.pth")
    
    print("\n✅ All plots generated.")
    print("✅ All episode data and summary metrics exported to CSVs.")
    print("✅ Model weights saved securely.")
    print("Training complete. You do not need to run this script again!")
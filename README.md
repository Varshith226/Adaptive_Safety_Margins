# Adaptive Safety Margins (ASM): A Curriculum Approach to Recovery RL

**Institution:** International Institute of Information Technology, Bangalore (IIIT-Bangalore)  
**Location:** Bengaluru, Karnataka  
**Roll Number:** BT2024227  

## Overview
Safe Reinforcement Learning (RL) requires agents to maximize task performance while satisfying hard safety constraints. Standard Recovery RL utilizes a fixed safety threshold ($\epsilon$) to switch between a task policy and a recovery policy. However, static thresholds force a trade-off: conservative values severely restrict exploration ("frozen robot" syndrome), while aggressive values lead to frequent safety violations.

This repository contains the code and implementation for **Adaptive Safety Margins (ASM)**. ASM dynamically schedules the risk threshold based on the agent's measured competence, smoothly transitioning from strict safety to relaxed exploration using a mathematically rigorous Sigmoid curriculum.

## Repository Structure
This project was developed through a dual-track methodology, culminating in a final merged architecture. The codebase is organized accordingly:

* `method1_proxy_env/`: Contains the theoretical prototype tested on a custom 2D proxy environment (`PointGoalEnv`). Validates the core ASM Sigmoid schedule against Linear and Step ablations using SGD-based action shielding.
* `method2_mujoco_decoupled/`: Contains the high-fidelity implementation on MuJoCo's `SafetyPointGoal1-v0`. Features a fully decoupled neural architecture with a dedicated Recovery Policy network and a Bellman-bootstrapped Safety Critic.
* `final_merged_system/`: The final, unified system combining the robust SAC optimization and hyperparameter calibration of Method 1 with the realistic physics environment of Method 2. 
* `RL_Report.pdf`: The comprehensive research paper detailing the theoretical proofs, experimental methodology, and final Pareto-optimal findings.

## Key Results
Evaluated on the continuous `SafetyPointGoal1-v0` environment, the merged ASM-Sigmoid agent successfully occupies the ideal region of the Safety-Return Pareto frontier. It achieves mean episodic returns comparable to an Aggressive baseline ($\epsilon=0.50$) while maintaining a cumulative constraint violation profile statistically identical to a strictly Conservative baseline ($\epsilon=0.05$).

## Installation & Setup
To run the final merged system and reproduce the results, ensure you have Python 3.10 installed, then build the isolated environment:

```bash
# 1. Create and activate a virtual environment
python -3.10 -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate

# 2. Install dependencies
pip install torch numpy pandas matplotlib safety-gymnasium
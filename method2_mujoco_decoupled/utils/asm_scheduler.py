import numpy as np

class AdaptiveSafetyMargin:
    def __init__(self, eps_min=0.1, eps_max=0.8, kappa=10, tau=0.5, alpha=0.1, burn_in_episodes=50):
        """
        Adaptive Safety Margin (ASM) Curriculum Scheduler.
        
        Args:
            eps_min: Strict safety floor (used during burn-in).
            eps_max: Maximum allowable risk at convergence.
            kappa: Transition speed of the sigmoid curve.
            tau: Competence inflection point (where rules start relaxing rapidly).
            alpha: Weight for the Exponentially Weighted Moving Average (EWMA).
            burn_in_episodes: Number of episodes to strictly enforce eps_min.
        """
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.kappa = kappa
        self.tau = tau
        self.alpha = alpha
        self.burn_in = burn_in_episodes
        
        self.rho = 0.0  # Initial competence is 0

    def update_competence(self, episode_success):
        """
        Updates the EWMA of the agent's task competence.
        episode_success should be 1 if the agent reached the goal safely, 0 otherwise.
        """
        self.rho = self.alpha * episode_success + (1 - self.alpha) * self.rho
        return self.rho

    def get_threshold(self, current_episode):
        """
        Calculates the dynamic threshold epsilon_t based on current competence.
        """
        # Enforce burn-in period (Student Driver mode)
        if current_episode <= self.burn_in:
            return self.eps_min
        
        # Calculate curriculum via Sigmoid function (Graduation mode)
        sigmoid_val = 1 / (1 + np.exp(-self.kappa * (self.rho - self.tau)))
        eps_t = self.eps_min + (self.eps_max - self.eps_min) * sigmoid_val
        
        return eps_t
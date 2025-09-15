from stable_baselines3 import PPO
import torch
import numpy as np

class FederatedTrainer:
    def __init__(self, num_locals, env_fn, ppo_kwargs, alpha=0.1):
        self.num_locals = num_locals
        self.alpha = alpha
        self.locals = [PPO('MlpPolicy', env_fn(), **ppo_kwargs) for _ in range(num_locals)]
        self.global_model = PPO('MlpPolicy', env_fn(), **ppo_kwargs)  # Dummy env

    def aggregate(self, scores):
        # Compute p_i (weights based on positive scores)
        pos_scores = [max(s, 0) for s in scores]
        total_pos = sum(pos_scores)
        if total_pos > 0:
            p = [s / total_pos for s in pos_scores]
        else:
            p = [1.0 / self.num_locals] * self.num_locals
        
        # Soft update aggregation
        global_sd = self.global_model.policy.state_dict()
        for key in global_sd:
            weighted_avg = sum(p[i] * self.locals[i].policy.state_dict()[key] for i in range(self.num_locals))
            global_sd[key] = self.alpha * global_sd[key] + (1 - self.alpha) * weighted_avg
        self.global_model.policy.load_state_dict(global_sd)
        
        # Distribute back
        for local in self.locals:
            local.policy.load_state_dict(global_sd)

    def train(self, total_timesteps, K_timesteps, model_path):
        rounds = total_timesteps // K_timesteps
        rewards = [[] for _ in range(self.num_locals)]
        for r in range(rounds):
            scores = []
            for i, local in enumerate(self.locals):
                local.learn(total_timesteps=K_timesteps)
                # Get mean episode reward from logger
                log_dict = local.logger.name_to_value
                last_reward = log_dict.get('rollout/ep_rew_mean', 0)
                scores.append(last_reward)
                rewards[i].append(last_reward)
            self.aggregate(scores)
        self.global_model.save(model_path)
        return rewards
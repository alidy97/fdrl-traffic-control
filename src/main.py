import argparse
from custom_env import CustomSumoEnv
from federated_trainer import FederatedTrainer
from utils import plot_reward_curves, evaluate_fixed_timing
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from src.wrapper import CustomObservationWrapper  # Add this import

NET_FILE = 'sumo_configs/single_intersection.net.xml'
ROU_FILE = 'sumo_configs/single_intersection.rou.xml'
EPISODES = 20  # Reduced for testing; set to 200 for full run
EPISODE_STEPS = 3600  # 1 hour sim
DELTA_TIME = 30
ACTIONS_PER_EP = EPISODE_STEPS // DELTA_TIME  # ~120
TOTAL_TIMESTEPS = EPISODES * ACTIONS_PER_EP  # ~2400 for test
K = 10  # Rounds per aggregation
K_TIMESTEPS = K * ACTIONS_PER_EP  # ~1200
PPO_KWARGS = {
    'learning_rate': 0.0001,
    'n_steps': ACTIONS_PER_EP,  # Buffer per ep
    'gamma': 0.9,
    'clip_range': 0.2,
    'verbose': 1
}

class RewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        mean_reward = self.model.logger.name_to_value.get('rollout/ep_rew_mean', 0)
        self.rewards.append(mean_reward)

def train_single():
    env = CustomObservationWrapper(CustomSumoEnv(NET_FILE, ROU_FILE, out_csv_name='results/single'))
    model = PPO('MlpPolicy', env, **PPO_KWARGS)
    model.set_logger(configure('results/single_log', ['csv']))
    callback = RewardCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    model.save('models/ppo_single.zip')
    return [callback.rewards]  # List of lists for consistency

def train_federated():
    env_fn = lambda: CustomObservationWrapper(CustomSumoEnv(NET_FILE, ROU_FILE, out_csv_name='results/fed'))
    trainer = FederatedTrainer(num_locals=3, env_fn=env_fn, ppo_kwargs=PPO_KWARGS)
    rewards = trainer.train(TOTAL_TIMESTEPS, K_TIMESTEPS, 'models/ppo_fed.zip')
    return rewards

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--federated', action='store_true')
    args = parser.parse_args()
    
    fixed_wait = evaluate_fixed_timing(NET_FILE, ROU_FILE)
    print(f'Fixed Timing Avg Wait: {fixed_wait}')
    
    if args.federated:
        rewards = train_federated()
        labels = ['Local 1', 'Local 2', 'Local 3']
    else:
        rewards = train_single()
        labels = ['Single PPO']
    
    plot_reward_curves(rewards, labels, 'results/reward_curves.png')
    print("Training complete! Check results/reward_curves.png for convergence plot.")
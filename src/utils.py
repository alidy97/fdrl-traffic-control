import matplotlib.pyplot as plt
import numpy as np
from custom_env import CustomSumoEnv
from src.wrapper import CustomObservationWrapper

def plot_reward_curves(rewards, labels, save_path):
    plt.figure()
    for r, label in zip(rewards, labels):
        plt.plot(np.cumsum(r) / np.arange(1, len(r)+1), label=label)  # Running avg
    plt.xlabel('Rounds/Episodes')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.title('Convergence Curves')
    plt.savefig(save_path)
    plt.close()

def evaluate_fixed_timing(net_file, rou_file, episodes=1):
    avg_waits = []
    for ep in range(episodes):
        env = CustomObservationWrapper(CustomSumoEnv(net_file, rou_file, out_csv_name=f'results/fixed_ep{ep}'))
        env.fixed_time = True  # Use sumo-rl's fixed timing mode
        obs, _ = env.reset()
        terminated = truncated = False
        total_wait = 0
        steps = 0
        while not (terminated or truncated):
            action = 0  # Ignored in fixed mode
            obs, reward, terminated, truncated, info = env.step(action)
            total_wait += env.get_average_waiting_time()
            steps += 1
        avg_waits.append(total_wait / steps if steps > 0 else 0)
        env.close()
    return np.mean(avg_waits)
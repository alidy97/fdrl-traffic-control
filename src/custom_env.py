import gymnasium as gym
import numpy as np
from sumo_rl.environment.env import SumoEnvironment
import traci
from gymnasium.spaces import Box

class CustomSumoEnv(SumoEnvironment):
    def __init__(self, net_file, route_file, out_csv_name=None, use_gui=False):
        super().__init__(
            net_file=net_file,
            route_file=route_file,
            out_csv_name=out_csv_name,
            use_gui=use_gui,
            delta_time=30,
            yellow_time=3,
            min_green=27,
            single_agent=True,
            sumo_warnings=False
        )
        self.prev_avg_wait = 0.0
        self.ts_id = list(self.traffic_signals.keys())[0]
        self.lanes = self.traffic_signals[self.ts_id].lanes
        print(f"Number of controlled lanes: {len(self.lanes)}")  # Debug: should be 8
        print(f"Observation space after init: {self.observation_space}")  # Debug the actual space

    def get_average_waiting_time(self):
        vehs = traci.vehicle.getIDList()
        if not vehs:
            return 0.0
        total_wait = sum(traci.vehicle.getAccumulatedWaitingTime(v) for v in vehs)
        return total_wait / len(vehs)

    def _compute_observations(self):
        state = []
        for lane in self.lanes:
            queue = traci.lane.getLastStepHaltingNumber(lane)  # Queue length
            state.append(queue)
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            wait = 0.0
            if veh_list:
                front_veh = veh_list[0]  # First is frontmost (closest to junction)
                wait = traci.vehicle.getAccumulatedWaitingTime(front_veh)
            state.append(wait)
        # Ensure 16 dimensions (8 queues + 8 waits), pad with 0 if less
        while len(state) < 16:
            state.append(0.0)
        return {self.ts_id: np.array(state, dtype=np.float32)}

    def _compute_rewards(self):
        current_avg_wait = self.get_average_waiting_time()
        reward = self.prev_avg_wait - current_avg_wait  # Delta avg wait
        self.prev_avg_wait = current_avg_wait
        return {self.ts_id: reward}

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.prev_avg_wait = self.get_average_waiting_time()
        print(f"Observation structure: {obs}")  # Debug: Check the structure of obs
        # Handle different observation structures
        if isinstance(obs, dict) and self.ts_id in obs:
            observation = obs[self.ts_id]
            if isinstance(observation, np.ndarray):
                return observation, info
            elif isinstance(observation, dict):
                for value in observation.values():
                    if isinstance(value, np.ndarray):
                        return value, info
        elif isinstance(obs, np.ndarray):
            return obs, info
        raise ValueError("Unexpected observation structure. Check obs: {obs}")

    def step(self, action):
        self.prev_avg_wait = self.get_average_waiting_time()  # Before step
        obs, reward, terminated, truncated, info = super().step(action)
        # Handle different observation structures
        if isinstance(obs, dict) and self.ts_id in obs:
            observation = obs[self.ts_id]
            if isinstance(observation, np.ndarray):
                obs_value = observation
            elif isinstance(observation, dict):
                for value in observation.values():
                    if isinstance(value, np.ndarray):
                        obs_value = value
                        break
                else:
                    raise ValueError("No valid observation array found in obs dictionary")
        elif isinstance(obs, np.ndarray):
            obs_value = obs
        else:
            raise ValueError("Unexpected observation structure in step. Check obs: {obs}")
        # Handle reward as float for single-agent
        reward_value = reward if isinstance(reward, (int, float)) else reward[self.ts_id]
        return obs_value, reward_value, terminated, truncated, info
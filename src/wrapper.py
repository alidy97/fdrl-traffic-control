from gymnasium import ObservationWrapper
import numpy as np
from gymnasium.spaces import Box

class CustomObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=np.inf, shape=(16,), dtype=np.float32)

    def observation(self, observation):
        if isinstance(observation, dict) and hasattr(self.env, 'ts_id') and self.env.ts_id in observation:
            obs = observation[self.env.ts_id]
            if len(obs) > 16:
                return obs[:16]  # Truncate if longer (unlikely)
            elif len(obs) < 16:
                return np.pad(obs, (0, 16 - len(obs)), 'constant')  # Pad with zeros if shorter
        return observation

    def get_average_waiting_time(self):
        # Delegate to the wrapped environment
        return self.env.get_average_waiting_time()
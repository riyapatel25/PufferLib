'''Picopark: Cooperative multiagent maze navigation.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.picopark import binding

# Multi-channel observation constants (must match picopark.h)
OBS_CHANNELS = 5  # block_color, agent_color, bullet, goal, key_door
OBS_EXTRAS = 11   # self_color, self_row, self_col, goal_col, camera_col, has_key, door_unlocked, door_row, key_row, key_col, key_holder_idx

class PicoPark(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, 
                 view_width=25, view_height=14, height=14, scroll_rate=3, num_agents=4,
                 buf=None, seed=0, frameskip=0):
        # Clamp num_agents to valid range (1-8)
        num_agents = max(1, min(num_agents, 8))
        
        # Multi-channel observation: [grid_ch0 | grid_ch1 | grid_ch2 | grid_ch3 | extras]
        # Channel values: block_color/agent_color 0-8, bullet/goal 0-1, extras 0-255
        grid_size = view_width * view_height
        obs_size = OBS_CHANNELS * grid_size + OBS_EXTRAS
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(obs_size,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(6)
        
        self.render_mode = render_mode
        self._num_agents_per_env = num_agents
        self.num_agents = num_envs * num_agents
        self.log_interval = log_interval
        self._num_envs = num_envs

        super().__init__(buf)
        
        c_envs = []
        for i in range(num_envs):
            start = i * num_agents
            end = (i + 1) * num_agents
            c_env = binding.env_init(
                self.observations[start:end],
                self.actions[start:end],
                self.rewards[start:end],
                self.terminals[start:end],
                self.truncations[start:end],
                seed + i,
                view_width=view_width,
                view_height=view_height,
                height=height,
                scroll_rate=scroll_rate,
                num_agents=num_agents
                # frameskip=frameskip
            )
            c_envs.append(c_env)
        
        self.c_envs = binding.vectorize(*c_envs)
 
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        c_envs = getattr(self, "c_envs", None)
        if c_envs is None:
            return
        binding.vec_close(c_envs)
        self.c_envs = None

if __name__ == '__main__':
    env = PicoPark(num_envs=1, render_mode='human', num_agents=8)
    env.reset()
    
    import time
    while True:
        actions = np.random.randint(0, 6, env.num_agents)
        obs, rewards, terminals, truncations, info = env.step(actions)
        env.render()
        time.sleep(0.1)
        
        if any(terminals):
            env.reset()

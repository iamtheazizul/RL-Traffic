import os
import sys
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import traci
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo',
    '-c', 'config/light.sumocfg',
    '--step-length', '0.1',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

class SumoEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Action: 0=stay, 1=switch
        self.action_space = spaces.Discrete(2)

        # Observation: queues on 6 detectors + current phase (7,)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7,), dtype=np.float32)

        # Phase durations in seconds (green, pedestrian, yellow)
        self.phase_durations = [37, 5, 3, 37, 5, 3]

        # Min green steps applies only on green phases 0 and 3
        self.min_green_steps = 30

        self.phase_step_counter = 0
        self.last_switch_step = -1000
        self.current_phase = 0

        self.max_steps = 1800
        self.step_count = 0

        self.cumulative_reward = 0.0
        self.total_queue = 0.0
        self.episode_count = 0

        self.episode_history = []
        self.reward_history = []
        self.queue_history = []

    def reset(self, seed=None, options=None):
        if traci.isLoaded():
            traci.close()

        traci.start(self.config)
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.total_queue = 0.0
        self.phase_step_counter = 0
        self.last_switch_step = -1000
        self.episode_count += 1

        self.current_phase = 0
        traci.trafficlight.setPhase("41896158", self.current_phase)
        traci.simulationStep()

        obs = self._get_state()
        return obs, {"episode": self.episode_count}

    def step(self, action):
        self.step_count += 1
        phase_duration = self.phase_durations[self.current_phase]
        time_in_phase = self.phase_step_counter
        apply_min_green = self.current_phase in [0, 3]

        can_switch = False
        if time_in_phase >= phase_duration:
            if apply_min_green:
                if (self.step_count - self.last_switch_step) >= self.min_green_steps:
                    if action == 1:
                        can_switch = True
            else:
                if action == 1:
                    can_switch = True

        if can_switch:
            self.current_phase = (self.current_phase + 1) % len(self.phase_durations)
            traci.trafficlight.setPhase("41896158", self.current_phase)
            self.last_switch_step = self.step_count
            self.phase_step_counter = 0
        else:
            # Stay in current phase
            pass

        traci.simulationStep()
        self.phase_step_counter += 1

        obs = self._get_state()
        reward = self._get_reward(obs)
        self.cumulative_reward += reward
        self.total_queue += np.sum(obs[:-1])

        terminated = False
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated

        info = {}
        if done:
            avg_queue = self.total_queue / self.step_count if self.step_count > 0 else 0
            self.episode_history.append(self.episode_count)
            self.reward_history.append(self.cumulative_reward)
            self.queue_history.append(avg_queue)
            info = {
                "episode": self.episode_count,
                "cumulative_reward": self.cumulative_reward,
                "avg_queue_length": avg_queue
            }
            print(f"Episode {self.episode_count} Summary: "
                  f"Cumulative Reward: {self.cumulative_reward:.2f}, "
                  f"Avg Queue Length: {avg_queue:.2f}")

        return obs, reward, terminated, truncated, info

    def _get_state(self):
        detectors = ["e2_2", "e2_3", "e2_4", "e2_6", "e2_11", "e2_9"]
        queues = []
        for det in detectors:
            try:
                q = traci.lanearea.getLastStepVehicleNumber(det)
            except Exception:
                q = 0
            queues.append(q)
        try:
            phase = traci.trafficlight.getPhase("41896158")
        except Exception:
            phase = 0
        return np.array(queues + [phase], dtype=np.float32)

    def _get_reward(self, obs):
        # Negative sum of queue lengths (excluding phase)
        return -float(np.sum(obs[:-1]))

    def close(self):
        if traci.isLoaded():
            traci.close()

    def render(self, mode='human'):
        pass  # No GUI

# Step 5: Custom Callback for Episode Control
class EpisodeCallback(BaseCallback):
    def __init__(self, env, total_episodes=30, verbose=0):
        super(EpisodeCallback, self).__init__(verbose)
        self.env = env
        self.total_episodes = total_episodes
        self.current_episode = 0

    def _on_step(self):
        if self.env.step_count >= self.env.max_steps:
            self.current_episode += 1
            if self.current_episode >= self.total_episodes:
                return False  # Stop training
        return True

# Step 6: Episode-based Training Loop with Stable Baselines3
print("\n=== Starting Episode-based Reinforcement Learning (DQN with Stable Baselines3) ===")

# Initialize environment
env = SumoEnv(Sumo_config)

# Check environment compatibility
from stable_baselines3.common.env_checker import check_env
check_env(env)

# Initialize DQN model
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.1,  # ALPHA
    gamma=0.9,          # GAMMA
    exploration_initial_eps=0.1,  # EPSILON
    exploration_final_eps=0.1,    # Constant exploration
    exploration_fraction=1.0,
    verbose=1,
    learning_starts=0,
    train_freq=1,
    batch_size=32,
    target_update_interval=1000
)

# Train for exactly 100 episodes
TOTAL_EPISODES = 500
callback = EpisodeCallback(env, total_episodes=TOTAL_EPISODES, verbose=1)
model.learn(total_timesteps=TOTAL_EPISODES * 1800, callback=callback, progress_bar=True)

# Save the model
model.save("dqn_sumo")

# Close the environment
env.close()

# Step 7: Visualization of Results
plt.figure(figsize=(10, 6))
plt.plot(env.episode_history, env.reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("RL Training (DQN): Cumulative Reward over Episodes")
plt.legend()
plt.grid(True)
plt.savefig("cumulative_reward_DQN.png")

plt.figure(figsize=(10, 6))
plt.plot(env.episode_history, env.queue_history, marker='o', linestyle='-', label="Average Queue Length")
plt.xlabel("Episode")
plt.ylabel("Average Queue Length")
plt.title("RL Training (DQN): Average Queue Length over Episodes")
plt.legend()
plt.grid(True)
plt.savefig("queue_length_DQN.png")
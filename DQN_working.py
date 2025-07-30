# Import necessary libraries
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

# Step 2: Establish path to SUMO
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Define SUMO configuration
Sumo_config = [
    'sumo',
    '-c', 'config/light.sumocfg',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

# Step 4: Define Custom SUMO Environment
class SumoEnv(gym.Env):
    def __init__(self, config):
        super(SumoEnv, self).__init__()
        self.config = config
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7,), dtype=np.float32)
        self.min_green_steps = 10
        self.min_yellow_steps = 5
        self.min_pedestrian_steps = 7
        self.step_count = 0
        self.max_steps = 1800
        self.cumulative_reward = 0.0
        self.total_queue = 0.0
        self.last_switch_step = -self.min_green_steps
        self.current_simulation_step = 0
        self.episode_count = 0
        self.episode_history = []
        self.reward_history = []
        self.queue_history = []

    def reset(self, seed=None, **kwargs):
        if traci.isLoaded():
            traci.close()
        traci.start(self.config)
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.total_queue = 0.0
        self.last_switch_step = -self.min_green_steps
        self.current_simulation_step = 0
        self.episode_count += 1
        state = self._get_state()
        info = {"episode": self.episode_count}
        return state, info

    def step(self, action):
        self.current_simulation_step = self.step_count
        self._apply_action(action)
        traci.simulationStep()
        new_state = self._get_state()
        reward = self._get_reward(new_state)
        self.cumulative_reward += reward
        self.total_queue += sum(new_state[:-1])
        self.step_count += 1

        terminated = False
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated

        info = {}
        if done:
            avg_queue = self.total_queue / self.step_count if self.step_count > 0 else 0
            self.episode_history.append(self.episode_count - 1)
            self.reward_history.append(self.cumulative_reward)
            self.queue_history.append(avg_queue)
            info = {
                "episode": self.episode_count,
                "cumulative_reward": self.cumulative_reward,
                "avg_queue_length": avg_queue
            }
            print(f"Episode {info['episode']} Summary: Cumulative Reward: {info['cumulative_reward']:.2f}, Avg Queue Length: {info['avg_queue_length']:.2f}")

        return new_state, reward, terminated, truncated, info

    def _get_state(self):
        detector_EB_0 = "e2_2"
        detector_SB_0 = "e2_3"
        detector_SB_1 = "e2_4"
        detector_WB_0 = "e2_6"
        detector_NB_0 = "e2_11"
        detector_NB_1 = "e2_9"
        traffic_light_id = "41896158"

        q_EB_0 = self._get_queue_length(detector_EB_0)
        q_SB_0 = self._get_queue_length(detector_SB_0)
        q_SB_1 = self._get_queue_length(detector_SB_1)
        q_WB_0 = self._get_queue_length(detector_WB_0)
        q_NB_0 = self._get_queue_length(detector_NB_0)
        q_NB_1 = self._get_queue_length(detector_NB_1)
        current_phase = self._get_current_phase(traffic_light_id)

        return np.array([q_EB_0, q_SB_0, q_SB_1, q_WB_0, q_NB_0, q_NB_1, current_phase], dtype=np.float32)
    
    # def _apply_action(self, action, tls_id="41896158"):
    #     if action == 0:
    #         return
    #     elif action == 1:
    #         if self.current_simulation_step - self.last_switch_step >= self.min_green_steps:
    #             current_phase = self._get_current_phase(tls_id)
    #             # Switch between phase 0 and phase 2 only
    #             next_phase = 2 if current_phase == 0 else 0
    #             traci.trafficlight.setPhase(tls_id, next_phase)
    #             self.last_switch_step = self.current_simulation_step

    # Apply the next action when prompted. Switching between one phase to another requires a buffer time.
    # **NEED TO FIX**
    def _apply_action(self, action, tls_id="41896158"):
        if action == 0:
            return
        elif action == 1:
            if self.current_simulation_step - self.last_switch_step >= self.min_green_steps:
                current_phase = self._get_current_phase(tls_id)
                try:
                    program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
                    num_phases = len(program.phases)
                    if num_phases == 0:
                        return
                    # Increment phase by 1 modulo total phases
                    next_phase = (current_phase + 1) % num_phases
                    traci.trafficlight.setPhase(tls_id, next_phase)
                    self.last_switch_step = self.current_simulation_step
                except traci.exceptions.TraCIException:
                    # Handle possible SUMO connection issues gracefully
                    pass

    # def _apply_action(self, action, tls_id="41896158"):
    #     if action == 0:
    #         return
    #     elif action == 1:
    #         current_phase = self._get_current_phase(tls_id)
    #         if current_phase in [0, 3]:
    #             if self.current_simulation_step - self.last_switch_step < self.min_green_steps:
    #                 return
    #         elif current_phase in [2, 5]:
    #             if self.current_simulation_step - self.last_switch_step < self.min_yellow_steps:
    #                 return
    #         elif current_phase in [1, 4]:
    #             if self.current_simulation_step - self.last_switch_step < self.min_pedestrian_steps:
    #                 return
    #         try:
    #             program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    #             num_phases = len(program.phases)
    #             if num_phases == 0:
    #                 return
    #             next_phase = (current_phase + 1) % num_phases
    #             traci.trafficlight.setPhase(tls_id, next_phase)
    #             self.last_switch_step = self.current_simulation_step
    #         except traci.exceptions.TraCIException as e:
    #             print(f"TraCIException during phase switch: {e}")

    def _get_reward(self, state):
        total_queue = sum(state[:-1])
        reward = -float(total_queue)
        return reward

    def _get_queue_length(self, detector_id):
        return traci.lanearea.getLastStepVehicleNumber(detector_id)

    def _get_current_phase(self, tls_id):
        return traci.trafficlight.getPhase(tls_id)

    def close(self):
        if traci.isLoaded():
            traci.close()

    def render(self, mode="human"):
        pass

# Step 5: Episode Callback with Robust Evaluation
class EpisodeCallbackDQN(BaseCallback):
    def __init__(self, env, total_episodes, eval_interval=25, eval_episodes=10, verbose=0):
        super(EpisodeCallbackDQN, self).__init__(verbose)
        self.env = env
        self.total_episodes = total_episodes
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.eval_episode_history = []
        self.eval_reward_history = []
        self.eval_queue_history = []

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        current_episode = self.env.episode_count
        # print(f"Callback triggered at episode {current_episode}")
        if current_episode % self.eval_interval == 0 and current_episode > 0:
            print(f"\n=== Starting Evaluation at Episode {current_episode} ===")
            eval_rewards = []
            eval_queues = []
            for eval_ep in range(self.eval_episodes):
                print(f"Running evaluation episode {eval_ep + 1}/{self.eval_episodes}")
                try:
                    state, _ = self.env.reset()
                    done = False
                    total_reward = 0.0
                    total_queue = 0.0
                    steps = 0
                    while not done:
                        action, _ = self.model.predict(state, deterministic=True)
                        state, reward, terminated, truncated, info = self.env.step(action)
                        total_reward += reward
                        total_queue += sum(state[:-1])
                        steps += 1
                        done = terminated or truncated
                    avg_queue = total_queue / steps if steps > 0 else 0
                    eval_rewards.append(total_reward)
                    eval_queues.append(avg_queue)
                    print(f"Evaluation Episode {eval_ep + 1}: Reward: {total_reward:.2f}, Avg Queue: {avg_queue:.2f}")
                except traci.exceptions.TraCIException as e:
                    print(f"TraCIException during evaluation episode {eval_ep + 1}: {e}")
                    continue
            if eval_rewards:  # Only store if evaluation was successful
                mean_reward = np.mean(eval_rewards)
                mean_queue = np.mean(eval_queues)
                self.eval_episode_history.append(current_episode)
                self.eval_reward_history.append(mean_reward)
                self.eval_queue_history.append(mean_queue)
                print(f"Evaluation Summary at Episode {current_episode}: Mean Reward: {mean_reward:.2f}, Mean Avg Queue: {mean_queue:.2f}")
            else:
                print(f"Evaluation at Episode {current_episode} failed due to errors")
        return True

# Step 6: Training Loop
print("\n=== Starting Episode-based Reinforcement Learning (DQN with Stable Baselines3) ===")

env = SumoEnv(Sumo_config)
from stable_baselines3.common.env_checker import check_env
check_env(env)

model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0001,  # Fixed: 10e-5 was too low, use 0.0001
    gamma=0.99,            # Increased from 0.9 to 0.99 for better long-term planning
    exploration_initial_eps=1,  # Start with 100% exploration
    exploration_final_eps=0.01,   # End with 1% exploration
    exploration_fraction=0.3,     # Decay over 30% of training
    verbose=1,
    learning_starts=5000,         # Start learning after 5000 steps
    train_freq=2,                 # Update every 2 steps (balanced)
    batch_size=128,                # Increased batch size
    target_update_interval=1000,   # Update target network more frequently
    buffer_size=100000,           # Larger replay buffer
    tau=0.01,                      # Soft target update
    gradient_steps=2              # Gradient steps per update matched with update frequency
)

TOTAL_EPISODES = 250  # For 10 evaluations (250 / 25 = 10)
callback = EpisodeCallbackDQN(env, total_episodes=TOTAL_EPISODES, eval_interval=25, eval_episodes=10)
model.learn(total_timesteps=TOTAL_EPISODES * env.max_steps, callback=callback, progress_bar=True)
model.save("dqn_sumo")

# Save training and evaluation metrics
np.save("dqn_episode_history.npy", np.array(env.episode_history))
np.save("dqn_reward_history.npy", np.array(env.reward_history))
np.save("dqn_queue_history.npy", np.array(env.queue_history))
np.save("dqn_eval_episode_history.npy", np.array(callback.eval_episode_history))
np.save("dqn_eval_reward_history.npy", np.array(callback.eval_reward_history))
np.save("dqn_eval_queue_history.npy", np.array(callback.eval_queue_history))

# Step 7: Plot Training and Evaluation Metrics
plt.figure(figsize=(12, 5))

# Plot Rewards
plt.subplot(1, 2, 1)
plt.plot(env.episode_history, env.reward_history, label='Training Reward', color='blue')
plt.plot(callback.eval_episode_history, callback.eval_reward_history, 'o-', label='Evaluation Reward', color='orange')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Training and Evaluation Rewards')
plt.legend()
plt.grid(True)

# Plot Queue Lengths
plt.subplot(1, 2, 2)
plt.plot(env.episode_history, env.queue_history, label='Training Avg Queue', color='blue')
plt.plot(callback.eval_episode_history, callback.eval_queue_history, 'o-', label='Evaluation Avg Queue', color='orange')
plt.xlabel('Episode')
plt.ylabel('Average Queue Length')
plt.title('Training and Evaluation Queue Lengths')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_evaluation_metrics.png')
plt.close()

env.close()
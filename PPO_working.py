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
from stable_baselines3 import PPO  # Changed from DQN to PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Define SUMO configuration
Sumo_config = [
    'sumo',
    '-c', 'config/ideal.sumocfg',
    '--step-length', '0.1',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

# Step 4: Define Custom SUMO Environment (Reusing your SumoEnv class)
class SumoEnv(gym.Env):
    def __init__(self, config):
        super(SumoEnv, self).__init__()
        self.config = config
        self.action_space = spaces.Discrete(2)  # 0 = keep phase, 1 = switch phase
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7,), dtype=np.float32)
        self.min_green_steps = 100
        self.min_yellow_steps = 30
        self.min_pedestrian_steps = 50
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
        self.braking_history = []

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
                    next_phase = (current_phase + 1) % num_phases
                    traci.trafficlight.setPhase(tls_id, next_phase)
                    self.last_switch_step = self.current_simulation_step
                except traci.exceptions.TraCIException:
                    pass

    # Uncomment the following method if you want to use a more complex action application logic
    # def _apply_action(self, action, tls_id="41896158"):
    #     if action == 0:
    #         return  # Keep current phase
    #     elif action == 1:
    #         current_phase = self._get_current_phase(tls_id)
    #         # Enforce minimum durations based on phase type
    #         if current_phase in [0, 3]:  # Primary green phases
    #             if self.current_simulation_step - self.last_switch_step < self.min_green_steps:
    #                 return
    #         elif current_phase in [2, 5]:  # Yellow phases
    #             if self.current_simulation_step - self.last_switch_step < self.min_yellow_steps:
    #                 return
    #         elif current_phase in [1, 4]:  # Pedestrian/transition green phases
    #             if self.current_simulation_step - self.last_switch_step < self.min_pedestrian_steps:
    #                 return
    #         try:
    #             program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    #             num_phases = len(program.phases)
    #             if num_phases == 0:
    #                 return
    #             next_phase = (current_phase + 1) % num_phases
    #             print(f"Step {self.current_simulation_step} (time {self.current_simulation_step * 0.1:.1f}s): Switching from phase {current_phase} to phase {next_phase}")
    #             traci.trafficlight.setPhase(tls_id, next_phase)
    #             self.last_switch_step = self.current_simulation_step
    #         except traci.exceptions.TraCIException as e:
    #             print(f"TraCIException during phase switch: {e}")
    #             pass

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

# Step 5: Custom Callback for Episode Control
class EpisodeCallbackPPO(BaseCallback):
    def __init__(self, env, total_episodes=400, eval_interval=50, eval_episodes=10, verbose=0):
        super(EpisodeCallbackPPO, self).__init__(verbose)
        self.env = env
        self.total_episodes = total_episodes
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.current_episode = 0
        
        # Lists to store evaluation metrics
        self.eval_episode_history = []
        self.eval_reward_history = []
        self.eval_queue_history = []

    def _evaluate_agent(self):
        eval_rewards = []
        eval_queues = []
        for _ in range(self.eval_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_queue = 0
            steps = 0
            while not done:
                action, _ = self.model.predict(state, deterministic=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_queue += sum(state[:-1])
                steps += 1
            avg_queue = episode_queue / steps if steps > 0 else 0
            eval_rewards.append(episode_reward)
            eval_queues.append(avg_queue)
        avg_eval_reward = np.mean(eval_rewards)
        avg_eval_queue = np.mean(eval_queues)
        print(f"[PPO] Evaluation after episode {self.current_episode}: Avg Reward: {avg_eval_reward:.2f}, Avg Queue Length: {avg_eval_queue:.2f}")

        # Store evaluation results
        self.eval_episode_history.append(self.current_episode)
        self.eval_reward_history.append(avg_eval_reward)
        self.eval_queue_history.append(avg_eval_queue)

        # Save evaluation results
        np.save("ppo_eval_episode_history.npy", np.array(self.eval_episode_history))
        np.save("ppo_eval_reward_history.npy", np.array(self.eval_reward_history))
        np.save("ppo_eval_queue_history.npy", np.array(self.eval_queue_history))

    def _on_step(self) -> bool:
        if self.env.step_count >= self.env.max_steps:
            self.current_episode += 1
            if self.current_episode % self.eval_interval == 0:
                self._evaluate_agent()
            if self.current_episode >= self.total_episodes:
                return False
        return True
# Step 6: Episode-based Training Loop with Stable Baselines3 (PPO)
print("\n=== Starting Episode-based Reinforcement Learning (PPO with Stable Baselines3) ===")

# Initialize environment
env = SumoEnv(Sumo_config)

# Check environment compatibility
check_env(env)

# Initialize PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.001,  # Suitable for PPO
    n_steps=2048,          # Number of steps per rollout
    batch_size=64,         # Mini-batch size
    n_epochs=10,           # Number of epochs per update
    gamma=0.95,            # Discount factor
    gae_lambda=0.95,       # GAE for advantage estimation
    clip_range=0.2,        # Clipping parameter for PPO
    verbose=1
)

# Train for exactly 100 episodes
TOTAL_EPISODES = 400
callback = EpisodeCallbackPPO(env, total_episodes=TOTAL_EPISODES)
model.learn(total_timesteps=TOTAL_EPISODES * env.max_steps, callback=callback, progress_bar=True)
# Save the model
model.save("ppo_sumo")

# For PPO (PPO_working.py)
np.save("ppo_episode_history.npy", env.episode_history)
np.save("ppo_reward_history.npy", env.reward_history)
np.save("ppo_queue_history.npy", env.queue_history)

# Close the environment
env.close()

# Step 7: Visualization of Results
# Plot Cumulative Reward
plt.figure(figsize=(10, 6))
plt.plot(env.episode_history, env.reward_history, marker='o', linestyle='-', label="Cumulative Reward", color='#1f77b4')
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("RL Training (PPO): Cumulative Reward over Episodes")
plt.legend()
plt.grid(True)
plt.savefig("cumulative_reward_PPO.png")

# Plot Average Queue Length
plt.figure(figsize=(10, 6))
plt.plot(env.episode_history, env.queue_history, marker='o', linestyle='-', label="Average Queue Length", color='#ff7f0e')
plt.xlabel("Episode")
plt.ylabel("Average Queue Length")
plt.title("RL Training (PPO): Average Queue Length over Episodes")
plt.legend()
plt.grid(True)
plt.savefig("queue_length_PPO.png")
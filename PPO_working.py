import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import traci
from stable_baselines3 import PPO
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
    '--delay', '1000',
    '--lateral-resolution', '0'
]

# Step 4: Define Custom SUMO Environment
class SumoEnv(gym.Env):
    def __init__(self, config):
        super(SumoEnv, self).__init__()
        self.config = config
        self.action_space = spaces.Discrete(2)  # 0 = keep phase, 1 = switch phase
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
        print(f"[Env] Resetting environment for episode {self.episode_count}")
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
            print(f"[Env] Episode {info['episode']} Summary: Cumulative Reward: {info['cumulative_reward']:.2f}, Avg Queue Length: {info['avg_queue_length']:.2f}")
        # else:
        #     print(f"[Env] Step {self.step_count}/{self.max_steps}: Action {action}, Reward {reward:.2f}")

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
    #                 print(f"[Env] Error: No phases found for TLS {tls_id}")
    #                 return
    #             next_phase = (current_phase + 1) % num_phases
    #             # print(f"[Env] Step {self.current_simulation_step} (time {self.current_simulation_step * 0.1:.1f}s): Switching from phase {current_phase} to phase {next_phase}")
    #             traci.trafficlight.setPhase(tls_id, next_phase)
    #             self.last_switch_step = self.current_simulation_step
    #         except traci.exceptions.TraCIException as e:
    #             print(f"[Env] TraCIException during phase switch: {e}")
    #             return

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
    def __init__(self, env, total_episodes, eval_interval=25, eval_episodes=10, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.total_episodes = total_episodes
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.eval_episode_history = []
        self.eval_reward_history = []
        self.eval_queue_history = []

    def _on_step(self):
        return True  # Required for continuation

    def _on_rollout_end(self):
        current_episode = self.env.episode_count
        print(f"[Callback] Rollout end at episode {current_episode}")
        if current_episode % self.eval_interval == 0 and current_episode > 0:
            self._evaluate_agent(current_episode)
        if current_episode >= self.total_episodes:
            print(f"[Callback] Total episodes {self.total_episodes} reached, final evaluation")
            self._evaluate_agent(current_episode)
        return True

    def _evaluate_agent(self, episode_number):
        print(f"[Callback] PPO Evaluation at episode {episode_number}")
        eval_rewards = []
        eval_queues = []
        for ep in range(self.eval_episodes):
            state, _ = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_queue = 0.0
            steps = 0
            while not done:
                action, _ = self.model.predict(state, deterministic=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_queue += sum(state[:-1])
                steps += 1
            avg_queue = ep_queue / steps if steps > 0 else 0
            eval_rewards.append(ep_reward)
            eval_queues.append(avg_queue)
        mean_r = np.mean(eval_rewards)
        mean_q = np.mean(eval_queues)
        self.eval_episode_history.append(episode_number)
        self.eval_reward_history.append(mean_r)
        self.eval_queue_history.append(mean_q)
        print(f"[Callback] Evaluation at episode {episode_number}: Mean Reward: {mean_r:.2f}, Mean Avg Queue: {mean_q:.2f}")

# Step 6: Training Loop 
print("\n=== Starting Episode-based RL (PPO with SB3) ===")
env = SumoEnv(Sumo_config)
check_env(env)
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0003,
    n_steps=env.max_steps,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,   # Like DQN
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)
TOTAL_EPISODES = 250
callback = EpisodeCallbackPPO(env, total_episodes=TOTAL_EPISODES, eval_interval=25, eval_episodes=10)
model.learn(total_timesteps=TOTAL_EPISODES * env.max_steps, callback=callback, progress_bar=True)
model.save("ppo_sumo")

# Save metrics
np.save("ppo_episode_history.npy", np.array(env.episode_history))
np.save("ppo_reward_history.npy", np.array(env.reward_history))
np.save("ppo_queue_history.npy", np.array(env.queue_history))
np.save("ppo_eval_episode_history.npy", np.array(callback.eval_episode_history))
np.save("ppo_eval_reward_history.npy", np.array(callback.eval_reward_history))
np.save("ppo_eval_queue_history.npy", np.array(callback.eval_queue_history))

env.close()

# Step 7: Side-by-side TRAINING + EVAL plots (like in your DQN)
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
plt.savefig('training_evaluation_metrics_ppo.png')
plt.close()
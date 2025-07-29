import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import traci

# Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Define SUMO configuration
Sumo_config = [
    'sumo',
    '-c', 'config/ideal.sumocfg',
    # '--step-length', '0.1',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

# Reinforcement Learning Hyperparameters
TOTAL_EPISODES = 250
STEPS_PER_EPISODE = 1800
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
ACTIONS = [0, 1]
Q_table = {}

# Lists to record data for plotting
episode_history = []
reward_history = []
queue_history = []

# Define Custom SUMO Environment (from DQN/PPO)
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

    # def _apply_action(self, action, tls_id="41896158"):
    #     if action == 0:
    #         return
    #     elif action == 1:
    #         if self.current_simulation_step - self.last_switch_step >= self.min_green_steps:
    #             current_phase = self._get_current_phase(tls_id)
    #             try:
    #                 program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    #                 num_phases = len(program.phases)
    #                 if num_phases == 0:
    #                     return
    #                 next_phase = (current_phase + 1) % num_phases
    #                 print(f"Step {self.current_simulation_step} (time {self.current_simulation_step * 0.1:.1f}s): Switching from phase {current_phase} to phase {next_phase}")
    #                 traci.trafficlight.setPhase(tls_id, next_phase)
    #                 self.last_switch_step = self.current_simulation_step
    #             except traci.exceptions.TraCIException as e:
    #                 print(f"TraCIException during phase switch: {e}")
    
    # Uncomment the following method if you want to use a more complex action application logic
    def _apply_action(self, action, tls_id="41896158"):
        if action == 0:
            return
        elif action == 1:
            current_phase = self._get_current_phase(tls_id)
            if current_phase in [0, 3]:
                if self.current_simulation_step - self.last_switch_step < self.min_green_steps:
                    return
            elif current_phase in [2, 5]:
                if self.current_simulation_step - self.last_switch_step < self.min_yellow_steps:
                    return
            elif current_phase in [1, 4]:
                if self.current_simulation_step - self.last_switch_step < self.min_pedestrian_steps:
                    return
            try:
                program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
                num_phases = len(program.phases)
                if num_phases == 0:
                    return
                next_phase = (current_phase + 1) % num_phases
                # print(f"Step {self.current_simulation_step} (time {self.current_simulation_step * 0.1:.1f}s): Switching from phase {current_phase} to phase {next_phase}")
                traci.trafficlight.setPhase(tls_id, next_phase)
                self.last_switch_step = self.current_simulation_step
            except traci.exceptions.TraCIException as e:
                print(f"TraCIException during phase switch: {e}")

    def _get_reward(self, state):
        total_queue = sum(state[:-1])
        return -float(total_queue)

    def _get_queue_length(self, detector_id):
        try:
            return traci.lanearea.getLastStepVehicleNumber(detector_id)
        except traci.exceptions.TraCIException:
            return 0.0

    def _get_current_phase(self, tls_id):
        try:
            return traci.trafficlight.getPhase(tls_id)
        except traci.exceptions.TraCIException:
            return 0

    def close(self):
        if traci.isLoaded():
            traci.close()

    def render(self, mode="human"):
        pass

# Q-learning Functions
def discretize_state(state):
    # Discretize queue lengths to manage Q-table size (e.g., bin into 0-5, 6-10, 11+ vehicles)
    bins = [0, 5, 10, np.inf]
    digitized = []
    for i in range(6):  # Queue lengths
        digitized.append(np.digitize(state[i], bins) - 1)
    digitized.append(int(state[6]))  # Current phase (0-5)
    return tuple(digitized)

def get_max_Q_value_of_state(s):
    if s not in Q_table:
        Q_table[s] = np.zeros(len(ACTIONS))
    return np.max(Q_table[s])

def update_Q_table(old_state, action, reward, new_state):
    old_state = discretize_state(old_state)
    new_state = discretize_state(new_state)
    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(len(ACTIONS))
    old_q = Q_table[old_state][action]
    best_future_q = get_max_Q_value_of_state(new_state)
    Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

def get_action_from_policy(state):
    state = discretize_state(state)
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        if state not in Q_table:
            Q_table[state] = np.zeros(len(ACTIONS))
        return int(np.argmax(Q_table[state]))

def evaluate_q_learning(env, Q_table, episodes=10, steps_per_episode=STEPS_PER_EPISODE):
    eval_rewards = []
    eval_queues = []
    for _ in range(episodes):
        state, _ = env.reset()
        cumulative_reward = 0.0
        total_queue = 0.0
        for step in range(steps_per_episode):
            state_discrete = discretize_state(state)
            if state_discrete not in Q_table:
                Q_table[state_discrete] = np.zeros(len(ACTIONS))
            action = int(np.argmax(Q_table[state_discrete]))
            state, reward, terminated, truncated, _ = env.step(action)
            cumulative_reward += reward
            total_queue += sum(state[:-1])
            if terminated or truncated:
                break
        avg_queue = total_queue / (step + 1) if step >= 0 else 0
        eval_rewards.append(cumulative_reward)
        eval_queues.append(avg_queue)
    avg_eval_reward = np.mean(eval_rewards)
    avg_eval_queue = np.mean(eval_queues)
    print(f"Q-learning evaluation: Avg Reward: {avg_eval_reward:.2f}, Avg Queue Length: {avg_eval_queue:.2f}")
    return avg_eval_reward, avg_eval_queue

# Main Q-learning training loop with periodic evaluation
print("\n=== Starting Episode-based Q-learning ===")
env = SumoEnv(Sumo_config)
eval_episode_history = []
eval_reward_history = []
eval_queue_history = []
EVAL_INTERVAL = 25
EVAL_EPISODES = 10

for episode in range(TOTAL_EPISODES):
    state, info = env.reset()
    cumulative_reward = 0.0
    total_queue = 0.0
    for step in range(STEPS_PER_EPISODE):
        action = get_action_from_policy(state)
        new_state, reward, terminated, truncated, info = env.step(action)
        update_Q_table(state, action, reward, new_state)
        cumulative_reward += reward
        total_queue += sum(new_state[:-1])
        state = new_state
        if terminated or truncated:
            break
    print(f"Episode {episode+1} Summary: Cumulative Reward: {env.reward_history[-1]:.2f}, Avg Queue Length: {env.queue_history[-1]:.2f}")
    # Evaluate every EVAL_INTERVAL episodes
    if (episode + 1) % EVAL_INTERVAL == 0:
        avg_eval_reward, avg_eval_queue = evaluate_q_learning(env, Q_table, episodes=EVAL_EPISODES)
        eval_episode_history.append(episode + 1)
        eval_reward_history.append(avg_eval_reward)
        eval_queue_history.append(avg_eval_queue)

# Save all histories (training and eval)
np.save("ql_episode_history.npy", np.array(env.episode_history))
np.save("ql_reward_history.npy", np.array(env.reward_history))
np.save("ql_queue_history.npy", np.array(env.queue_history))
np.save("ql_eval_episode_history.npy", np.array(eval_episode_history))
np.save("ql_eval_reward_history.npy", np.array(eval_reward_history))
np.save("ql_eval_queue_history.npy", np.array(eval_queue_history))
import pickle
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q_table, f)
env.close()

# Step 7: Side-by-side TRAINING + EVAL plots (as DQN/PPO)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(env.episode_history, env.reward_history, label='Training Reward', color='blue')
plt.plot(eval_episode_history, eval_reward_history, 'o-', label='Evaluation Reward', color='orange')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Training and Evaluation Rewards')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(env.episode_history, env.queue_history, label='Training Avg Queue', color='blue')
plt.plot(eval_episode_history, eval_queue_history, 'o-', label='Evaluation Avg Queue', color='orange')
plt.xlabel('Episode')
plt.ylabel('Average Queue Length')
plt.title('Training and Evaluation Queue Lengths')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_evaluation_metrics_ql.png')
plt.close()
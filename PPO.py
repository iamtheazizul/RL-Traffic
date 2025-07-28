import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import traci
import time
from datetime import datetime

# Establish path to SUMO
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# SUMO configuration
Sumo_config = [
    'sumo-gui',
    '-c', 'config/light.sumocfg',
    '--step-length', '1',
    '--delay', '10'
]

# Hyperparameters
TOTAL_EPISODES = 50
STEPS_PER_EPISODE = 1500
MIN_GREEN_SECONDS = 10
YELLOW_SECONDS = 3


class SumoEnv(gym.Env):
    def __init__(self):
        super(SumoEnv, self).__init__()
        self.action_space = spaces.Discrete(2)  # 0 = keep phase, 1 = switch phase
        self.observation_space = spaces.Box(
            low=np.zeros(7, dtype=np.float32),
            high=np.full(7, np.inf, dtype=np.float32),
            dtype=np.float32
        )
        self.tls_id = "41896158"
        # Convert seconds to simulation steps
        self.min_green_steps = int(MIN_GREEN_SECONDS)  # 420 steps
        self.yellow_duration = int(YELLOW_SECONDS)    # 30 steps
        self.last_switch_step = -self.min_green_steps

        self.phase_start_step = 0
        self.current_phase = 0

        self.current_simulation_step = 0
        self.total_steps = STEPS_PER_EPISODE

        self.connection_active = False
        self.connection_label = "default"
        self.valid_ids = False
        self.max_retries = 5
        self.retry_delay = 2

    def reset(self, seed=None, options=None, retry_count=0):
        if self.connection_active:
            try:
                traci.close()
                time.sleep(self.retry_delay)
            except traci.exceptions.TraCIException:
                pass
            finally:
                self.connection_active = False

        try:
            self.connection_label = f"sumo_{id(self)}_{retry_count}"
            traci.start(Sumo_config, label=self.connection_label)
            self.connection_active = True
            traci.simulationStep()
            self._validate_ids()
        except traci.exceptions.TraCIException as e:
            if retry_count < self.max_retries:
                time.sleep(self.retry_delay)
                return self.reset(seed, options, retry_count + 1)
            else:
                raise e

        self.current_simulation_step = 0
        self.last_switch_step = -self.min_green_steps
        self.phase_start_step = 0
        self.current_phase = 0
        traci.trafficlight.setPhase(self.tls_id, 0)  # start with phase 0 green

        state = self._get_state()
        return state, {}

    def _validate_ids(self):
        try:
            traffic_lights = traci.trafficlight.getIDList()
            detectors = traci.lanearea.getIDList()
            expected_detectors = [
                "e2_2", "e2_3", "e2_4",
                "e2_6", "e2_11", "e2_9"
            ]
            if self.tls_id not in traffic_lights:
                self.valid_ids = False
                return
            if not all(d in detectors for d in expected_detectors):
                self.valid_ids = False
                return
            self.valid_ids = True
        except traci.exceptions.TraCIException:
            self.valid_ids = False

    def step(self, action):
        self.current_simulation_step += 1
        self._apply_action(action)

        try:
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            vehicle_count = traci.vehicle.getIDCount()
            if current_time < 0:
                return self._get_state(), 0.0, True, False, {"error": "Negative simulation time"}
            if vehicle_count == 0 and self.current_simulation_step < self.total_steps:
                return self._get_state(), 0.0, True, False, {"error": "No vehicles"}
        except traci.exceptions.TraCIException as e:
            return self._get_state(), 0.0, True, False, {"error": str(e)}

        new_state = self._get_state()
        reward = self._get_reward(new_state)
        done = self.current_simulation_step >= self.total_steps
        truncated = False
        info = {}
        return new_state, reward, done, truncated, info

    def _get_state(self):
        if not self.valid_ids:
            return np.zeros(7, dtype=np.float32)
        try:
            q_EB_0 = self._get_queue_length("e2_2")
            q_SB_0 = self._get_queue_length("e2_3")
            q_SB_1 = self._get_queue_length("e2_4")
            q_WB_0 = self._get_queue_length("e2_6")
            q_NB_0 = self._get_queue_length("e2_11")
            q_NB_1 = self._get_queue_length("e2_9")
            current_phase = self._get_current_phase(self.tls_id)
            state = np.array([q_EB_0, q_SB_0, q_SB_1, q_WB_0, q_NB_0, q_NB_1, current_phase], dtype=np.float32)
            return state
        except traci.exceptions.TraCIException:
            return np.zeros(7, dtype=np.float32)

    def _get_reward(self, state):
        total_queue = sum(state[:-1])
        return -float(total_queue)

    # def _apply_action(self, action):
    #     if not self.valid_ids:
    #         return

    #     steps_in_phase = self.current_simulation_step - self.phase_start_step
    #     current_phase = self._get_current_phase(self.tls_id)

    #     # Logic: must respect green and yellow durations, switch only when allowed
    #     if current_phase in [0, 2]:  # green phases
    #         if action == 1 and steps_in_phase >= self.min_green_steps:
    #             # switch to yellow after green
    #             next_yellow = 1 if current_phase == 0 else 3
    #             traci.trafficlight.setPhase(self.tls_id, next_yellow)
    #             self.phase_start_step = self.current_simulation_step
    #             self.current_phase = next_yellow
    #     elif current_phase in [1, 3]:  # yellow phases
    #         if steps_in_phase >= self.yellow_duration:
    #             # switch to opposite green after yellow
    #             next_green = 2 if current_phase == 1 else 0
    #             traci.trafficlight.setPhase(self.tls_id, next_green)
    #             self.phase_start_step = self.current_simulation_step
    #             self.current_phase = next_green
    #     else:
    #         # fallback safety
    #         traci.trafficlight.setPhase(self.tls_id, 0)
    #         self.phase_start_step = self.current_simulation_step
    #         self.current_phase = 0

    def _apply_action(self, action):
        # logger.debug(f"Applying action: {action}")
        if not self.valid_ids:
            return
        if action == 0:
            return
        elif action == 1:
            if self.current_simulation_step - self.last_switch_step >= self.min_green_steps:
                try:
                    program = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
                    num_phases = len(program.phases)
                    if num_phases == 0:
                        return
                    next_phase = (self._get_current_phase(self.tls_id) + 1) % num_phases
                    traci.trafficlight.setPhase(self.tls_id, next_phase)
                    self.last_switch_step = self.current_simulation_step
                    # logger.debug(f"Switched to phase {next_phase}")
                except traci.exceptions.TraCIException as e:
                    return

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
        if self.connection_active:
            try:
                traci.close()
                self.connection_active = False
            except traci.exceptions.TraCIException:
                pass


def main():
    env = SumoEnv()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        verbose=1
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("results", timestamp)
    os.makedirs(result_dir, exist_ok=True)

    episode_history = []
    reward_history = []
    queue_history = []

    print("\n=== Starting PPO Training ===")
    for episode in range(TOTAL_EPISODES):
        obs, _ = env.reset()
        cumulative_reward = 0.0
        total_queue = 0.0
        print(f"\n=== Episode {episode + 1}/{TOTAL_EPISODES} ===")
        try:
            for step in range(STEPS_PER_EPISODE):
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = env.step(action)
                cumulative_reward += reward
                total_queue += sum(obs[:-1])
                if done or truncated or step == STEPS_PER_EPISODE - 1:
                    break
                if step % 100 == 0:
                    print(f"Step {step}/{STEPS_PER_EPISODE}, Action: {action}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}")
            model.learn(total_timesteps=STEPS_PER_EPISODE, reset_num_timesteps=False)
        except traci.exceptions.TraCIException as e:
            print(f"Episode {episode + 1} failed: {e}")
            env.close()
            sys.exit(1)
        episode_history.append(episode)
        reward_history.append(cumulative_reward)
        queue_history.append(total_queue / env.current_simulation_step if env.current_simulation_step > 0 else 0)
        print(f"Episode {episode + 1} Summary: Cumulative Reward: {cumulative_reward:.2f}, Avg Queue Length: {queue_history[-1]:.2f}")

    env.close()
    print("\nPPO Training completed.")

    # Save plots
    plt.figure(figsize=(10, 6))
    plt.plot(episode_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("RL Training (PPO): Cumulative Reward over Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "cumulative_reward_ppo.png"))

    plt.figure(figsize=(10, 6))
    plt.plot(episode_history, queue_history, marker='o', linestyle='-', label="Average Queue Length")
    plt.xlabel("Episode")
    plt.ylabel("Average Queue Length")
    plt.title("RL Training (PPO): Average Queue Length over Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "queue_length_ppo.png"))

    print(f"Plots saved in: {result_dir}")


if __name__ == "__main__":
    main()
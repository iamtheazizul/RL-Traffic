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
    '--step-length', '0.1',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

# Define Fixed-Time SUMO Environment with Custom Phase Durations
class SumoEnvFixedTime(gym.Env):
    def __init__(self, config):
        super(SumoEnvFixedTime, self).__init__()
        self.config = config
        self.action_space = spaces.Discrete(2)  # Keep for compatibility, but action ignored
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7,), dtype=np.float32)
        self.phase_durations = [37, 5, 3, 37, 5, 3]  # Durations in seconds
        self.phase_durations_steps = [int(d / 0.1) for d in self.phase_durations]  # Convert to steps
        self.current_phase_index = 0
        self.step_count = 0
        self.max_steps = 1800  # Steps per episode - converging to full two cycles
        self.cumulative_reward = 0.0
        self.total_queue = 0.0
        self.last_switch_step = 0
        self.current_simulation_step = 0
        self.episode_count = 0

        # Lists to record data for plotting
        self.episode_history = []
        self.reward_history = []
        self.queue_history = []

    # Each time an episode is done, we reset the environment
    # Here I manually define the traffic light phases to avoid any simulation mismatch.
    def reset(self, seed=None, **kwargs):
        if traci.isLoaded():
            traci.close()
        traci.start(self.config)
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.total_queue = 0.0
        self.last_switch_step = 0
        self.current_simulation_step = 0
        self.current_phase_index = 0
        self.episode_count += 1

        # Define and set the fixed-time traffic light program
        tls_id = "41896158"
        logic = traci.trafficlight.Logic(
            programID="fixed_time_custom",
            type=0,  # Static
            currentPhaseIndex=0,
            phases=[
                traci.trafficlight.Phase(duration=37.0, state="gGGggrrrgGGgrrrrGrG", minDur=37.0, maxDur=37.0),
                traci.trafficlight.Phase(duration=5.0, state="gGGggrrrgGGgrrrrrrr", minDur=5.0, maxDur=5.0),
                traci.trafficlight.Phase(duration=3.0, state="yyyyyrrryyyyrrrrrrr", minDur=3.0, maxDur=3.0),
                traci.trafficlight.Phase(duration=37.0, state="rrrrrgGgrrrrgGgGrGr", minDur=37.0, maxDur=37.0),
                traci.trafficlight.Phase(duration=5.0, state="rrrrrgGgrrrrgGgrrrr", minDur=5.0, maxDur=5.0),
                traci.trafficlight.Phase(duration=3.0, state="rrrrryyyrrrryyyrrrr", minDur=3.0, maxDur=3.0)
            ]
        )
        try:
            traci.trafficlight.setProgramLogic(tls_id, logic)
            traci.trafficlight.setProgram(tls_id, "fixed_time_custom")
            traci.trafficlight.setPhase(tls_id, 0)
            print(f"Episode {self.episode_count}: Initialized to phase 0 at step 0 with custom program")
        except traci.exceptions.TraCIException as e:
            print(f"Error setting traffic light program: {e}")
            sys.exit("Failed to set traffic light program")

        # Validate applied program
        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        applied_durations = [phase.duration for phase in program.phases]
        print(f"Applied phase durations: {applied_durations}")
        if applied_durations != self.phase_durations:
            print(f"Warning: Applied durations {applied_durations} do not match expected {self.phase_durations}")

        state = self._get_state()
        info = {"episode": self.episode_count}
        return state, info

    # Functions where a simualtion step is taken, action prompted and reward calculated
    # Computes cumulative reward and avg. queue length
    def step(self, action=None):
        self.current_simulation_step = self.step_count
        self._apply_fixed_time_action()
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

    # Get state values from detectors
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

    # Apply the next action when prompted. Follows the fixed time pattern.
    def _apply_fixed_time_action(self, tls_id="41896158"):
        current_phase_duration = self.phase_durations_steps[self.current_phase_index]
        if self.current_simulation_step >= self.last_switch_step + current_phase_duration:
            num_phases = len(self.phase_durations)
            self.current_phase_index = (self.current_phase_index + 1) % num_phases
            try:
                traci.trafficlight.setPhase(tls_id, self.current_phase_index)
                self.last_switch_step = self.current_simulation_step
                print(f"Step {self.current_simulation_step} (time {self.current_simulation_step * 0.1:.1f}s): Switching to phase {self.current_phase_index} (duration {self.phase_durations[self.current_phase_index]}s)")
            except traci.exceptions.TraCIException as e:
                print(f"Error switching to phase {self.current_phase_index}: {e}")
        # Log current phase and time to detect premature switches
        current_phase = self._get_current_phase(tls_id)
        if current_phase != self.current_phase_index:
            print(f"Warning: At step {self.current_simulation_step} (time {self.current_simulation_step * 0.1:.1f}s), SUMO phase {current_phase} does not match expected phase {self.current_phase_index}")

    # Reward function computed using the queue length
    def _get_reward(self, state):
        total_queue = sum(state[:-1])
        return -float(total_queue)

    # TraCI call for queue data
    def _get_queue_length(self, detector_id):
        try:
            return traci.lanearea.getLastStepVehicleNumber(detector_id)
        except traci.exceptions.TraCIException:
            return 0.0
    
    # TraCI call for phase data
    def _get_current_phase(self, tls_id):
        try:
            return traci.trafficlight.getPhase(tls_id)
        except traci.exceptions.TraCIException:
            return 0

    # TraCI call for closing
    def close(self):
        if traci.isLoaded():
            traci.close()

    # For non-GUI mode
    def render(self, mode="human"):
        pass

# Main execution
print("\n=== Starting Fixed-Time Traffic Light Simulation with Corrected Phases ===")

# Initialize environment
env = SumoEnvFixedTime(Sumo_config)

# Validate number of phases
traci.start(Sumo_config)
try:
    program = traci.trafficlight.getAllProgramLogics("41896158")[0]
    num_phases = len(program.phases)
    applied_durations = [phase.duration for phase in program.phases]
    print(f"SUMO configuration has {num_phases} phases with durations: {applied_durations}")
    if num_phases != len(env.phase_durations):
        print(f"Warning: SUMO has {num_phases} phases, but code expects {len(env.phase_durations)}")
except traci.exceptions.TraCIException as e:
    print(f"Error validating phases: {e}")
traci.close()

# Run for 100 episodes
TOTAL_EPISODES = 400
for episode in range(TOTAL_EPISODES):
    state, info = env.reset()
    for step in range(env.max_steps):
        state, reward, terminated, truncated, info = env.step()
        if terminated or truncated:
            break

# For FT (FT_working.py)
np.save("ft_episode_history.npy", env.episode_history)
np.save("ft_reward_history.npy", env.reward_history)
np.save("ft_queue_history.npy", env.queue_history)

# Close the environment
env.close()

# Visualization of Results
plt.figure(figsize=(10, 6))
plt.plot(env.episode_history, env.reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Fixed-Time Control: Cumulative Reward over Episodes")
plt.legend()
plt.grid(True)
plt.savefig("cumulative_reward_fixed_time_corrected.png")

plt.figure(figsize=(10, 6))
plt.plot(env.episode_history, env.queue_history, marker='o', linestyle='-', label="Average Queue Length")
plt.xlabel("Episode")
plt.ylabel("Average Queue Length")
plt.title("Fixed-Time Control: Average Queue Length over Episodes")
plt.legend()
plt.grid(True)
plt.savefig("queue_length_fixed_time_corrected.png")
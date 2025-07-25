import os
import sys
import random
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

# SUMO imports
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")
import traci

from datetime import datetime

# SUMO configuration
Sumo_config = [
    'sumo',
    '-c', 'osm_25.sumocfg',
    '--step-length', '0.10',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

# RL / simulation params
TOTAL_EPISODES = 400
STEPS_PER_EPISODE = 1500

# Logging / storage
episode_rewards = []
episode_avg_queues = []
episode_history = []

# Detector IDs and TLS ID
detectors_EB = ["e2_2"]
detectors_SB = ["e2_3", "e2_4"]
detectors_WB = ["e2_6"]
detectors_NB = ["e2_11", "e2_9"]

tls_id = "41896158"

def get_queue_length(detector_id):
    try:
        return traci.lanearea.getLastStepVehicleNumber(detector_id)
    except traci.exceptions.TraCIException:
        return 0.0

def get_current_phase(tls_id):
    try:
        return traci.trafficlight.getPhase(tls_id)
    except traci.exceptions.TraCIException:
        return 0

def get_state():
    q_EB = [get_queue_length(det) for det in detectors_EB]
    q_SB = [get_queue_length(det) for det in detectors_SB]
    q_WB = [get_queue_length(det) for det in detectors_WB]
    q_NB = [get_queue_length(det) for det in detectors_NB]

    phase = get_current_phase(tls_id)
    return tuple(q_EB + q_SB + q_WB + q_NB +[phase])

def get_reward(state):
    total_queue = sum(state[:-1])
    return -float(total_queue)

def apply_fixed_time_policy(step, tls_id="41896158"):
    cycle_length = 2 * (42 + 3)  # 2 groups each with green+yellow
    cycle_length_steps = int(cycle_length / 0.1)  # Convert seconds to simulation steps

    green_duration = int(42 / 0.1)  # 420 steps
    yellow_duration = int(3 / 0.1)  # 30 steps

    step_in_cycle = step % cycle_length_steps

    if step_in_cycle < green_duration:
        phase_to_set = 0  # Green phase 0
    elif step_in_cycle < green_duration + yellow_duration:
        phase_to_set = 1  # Yellow phase 1
    elif step_in_cycle < green_duration + yellow_duration + green_duration:
        phase_to_set = 2  # Green phase 2
    else:
        phase_to_set = 3  # Yellow phase 3

    current_phase = traci.trafficlight.getPhase(tls_id)
    if current_phase != phase_to_set:
        traci.trafficlight.setPhase(tls_id, phase_to_set)

def run_episode(episode_num):
    print(f"\n--- Starting episode {episode_num+1}/{TOTAL_EPISODES} ---")
    traci.start(Sumo_config)
    # traci.gui.setSchema("View #0", "real world")

    cumulative_reward = 0.0
    cumulative_queue = 0.0

    for step in range(STEPS_PER_EPISODE):
        state = get_state()

        apply_fixed_time_policy(step)

        traci.simulationStep()

        new_state = get_state()
        reward = get_reward(new_state)

        cumulative_reward += reward
        cumulative_queue += sum(new_state[:-1])

        if step % 100 == 0:
            print(f"Step {step}: Reward={reward:.2f}, Cumulative Reward={cumulative_reward:.2f}")

    avg_queue = cumulative_queue / STEPS_PER_EPISODE
    print(f"Episode {episode_num+1} finished. Total Reward: {cumulative_reward:.2f}, Avg Queue: {avg_queue:.2f}")

    traci.close()

    return cumulative_reward, avg_queue

def main():
    for ep in range(TOTAL_EPISODES):
        reward, avg_queue = run_episode(ep)
        episode_rewards.append(reward)
        episode_avg_queues.append(avg_queue)
        episode_history.append(ep+1)

    # Save plots in a 'plots' folder with timestamps:
    output_dir = "ft"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reward_plot_path = os.path.join(output_dir, f"cumulative_reward_{timestamp}.png")
    queue_plot_path = os.path.join(output_dir, f"avg_queue_{timestamp}.png")

    # Plot cumulative rewards
    plt.figure(figsize=(10, 6))
    plt.plot(episode_history, episode_rewards, marker='o', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Fixed-Time Policy: Cumulative Reward per Episode")
    plt.grid(True)
    plt.savefig(reward_plot_path)

    # Plot average queue length
    plt.figure(figsize=(10, 6))
    plt.plot(episode_history, episode_avg_queues, marker='o', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Average Queue Length")
    plt.title("Fixed-Time Policy: Average Queue Length per Episode")
    plt.grid(True)
    plt.savefig(queue_plot_path)

    print(f"Plots saved to:\n{reward_plot_path}\n{queue_plot_path}")

if __name__ == "__main__":
    main()
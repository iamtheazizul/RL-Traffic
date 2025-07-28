import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load data from each run
try:
    ql_episode_history = np.load("ql_episode_history.npy")
    ql_reward_history = np.load("ql_reward_history.npy")
    ql_queue_history = np.load("ql_queue_history.npy")
    dqn_episode_history = np.load("dqn_episode_history.npy")
    dqn_reward_history = np.load("dqn_reward_history.npy")
    dqn_queue_history = np.load("dqn_queue_history.npy")
    ppo_episode_history = np.load("ppo_episode_history.npy")
    ppo_reward_history = np.load("ppo_reward_history.npy")
    ppo_queue_history = np.load("ppo_queue_history.npy")
    ft_episode_history = np.load("ft_episode_history.npy")
    ft_reward_history = np.load("ft_reward_history.npy")
    ft_queue_history = np.load("ft_queue_history.npy")
except FileNotFoundError as e:
    print(f"Error: Data file not found: {e}. Please run all algorithms and save history data.")
    exit(1)

# Truncate to 20 episodes for fair comparison (since FT has 20 episodes)
max_episodes = min(
    len(ql_episode_history), len(dqn_episode_history),
    len(ppo_episode_history), len(ft_episode_history)
)
ql_episode_history = ql_episode_history[:max_episodes]
ql_reward_history = ql_reward_history[:max_episodes]
ql_queue_history = ql_queue_history[:max_episodes]
dqn_episode_history = dqn_episode_history[:max_episodes]
dqn_reward_history = dqn_reward_history[:max_episodes]
dqn_queue_history = dqn_queue_history[:max_episodes]
ppo_episode_history = ppo_episode_history[:max_episodes]
ppo_reward_history = ppo_reward_history[:max_episodes]
ppo_queue_history = ppo_queue_history[:max_episodes]
ft_episode_history = ft_episode_history[:max_episodes]
ft_reward_history = ft_reward_history[:max_episodes]
ft_queue_history = ft_queue_history[:max_episodes]

# Determine best algorithm for cumulative reward (highest average)
reward_avgs = {
    "Q-learning": np.mean(ql_reward_history),
    "DQN": np.mean(dqn_reward_history),
    "PPO": np.mean(ppo_reward_history),
    "Fixed-Time": np.mean(ft_reward_history)
}
best_reward_algo = max(reward_avgs, key=reward_avgs.get)
print(f"Best algorithm for cumulative reward: {best_reward_algo} (Avg: {reward_avgs[best_reward_algo]:.2f})")

# Determine best algorithm for queue length (lowest average)
queue_avgs = {
    "Q-learning": np.mean(ql_queue_history),
    "DQN": np.mean(dqn_queue_history),
    "PPO": np.mean(ppo_queue_history),
    "Fixed-Time": np.mean(ft_queue_history)
}
best_queue_algo = min(queue_avgs, key=queue_avgs.get)
print(f"Best algorithm for average queue length: {best_queue_algo} (Avg: {queue_avgs[best_queue_algo]:.2f})")

# Plot settings
colors = {
    "Q-learning": '#9467bd',
    "DQN": '#1f77b4',
    "PPO": '#2ca02c',
    "Fixed-Time": '#ff7f0e'
}
markers = {
    "Q-learning": 'd',
    "DQN": 'o',
    "PPO": '^',
    "Fixed-Time": 's'
}
linewidths = {
    algo: 3 if algo == best_reward_algo else 1 for algo in reward_avgs
}

# Combined Cumulative Reward Plot
plt.figure(figsize=(14, 8))
plt.plot(ql_episode_history, ql_reward_history, marker=markers["Q-learning"], linestyle='-', 
         label="Q-learning Cumulative Reward", color=colors["Q-learning"], 
         linewidth=linewidths["Q-learning"])
plt.plot(dqn_episode_history, dqn_reward_history, marker=markers["DQN"], linestyle='-', 
         label="DQN Cumulative Reward", color=colors["DQN"], 
         linewidth=linewidths["DQN"])
plt.plot(ppo_episode_history, ppo_reward_history, marker=markers["PPO"], linestyle='-', 
         label="PPO Cumulative Reward", color=colors["PPO"], 
         linewidth=linewidths["PPO"])
plt.plot(ft_episode_history, ft_reward_history, marker=markers["Fixed-Time"], linestyle='--', 
         label="Fixed-Time Cumulative Reward", color=colors["Fixed-Time"], 
         linewidth=linewidths["Fixed-Time"])
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Comparison: Cumulative Reward over Episodes")
plt.legend()
plt.grid(True)
# Annotate the best algorithm
plt.annotate(f'Best: {best_reward_algo} (Avg: {reward_avgs[best_reward_algo]:.2f})',
             xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
             bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))
plt.savefig("comparison_cumulative_reward.png")
plt.close()

# Update linewidths for queue length
linewidths = {
    algo: 3 if algo == best_queue_algo else 1 for algo in queue_avgs
}

# Combined Average Queue Length Plot
plt.figure(figsize=(14, 8))
plt.plot(ql_episode_history, ql_queue_history, marker=markers["Q-learning"], linestyle='-', 
         label="Q-learning Avg Queue Length", color=colors["Q-learning"], 
         linewidth=linewidths["Q-learning"])
plt.plot(dqn_episode_history, dqn_queue_history, marker=markers["DQN"], linestyle='-', 
         label="DQN Avg Queue Length", color=colors["DQN"], 
         linewidth=linewidths["DQN"])
plt.plot(ppo_episode_history, ppo_queue_history, marker=markers["PPO"], linestyle='-', 
         label="PPO Avg Queue Length", color=colors["PPO"], 
         linewidth=linewidths["PPO"])
plt.plot(ft_episode_history, ft_queue_history, marker=markers["Fixed-Time"], linestyle='--', 
         label="Fixed-Time Avg Queue Length", color=colors["Fixed-Time"], 
         linewidth=linewidths["Fixed-Time"])
plt.xlabel("Episode")
plt.ylabel("Average Queue Length")
plt.title("Comparison: Average Queue Length over Episodes")
plt.legend()
plt.grid(True)
# Annotate the best algorithm
plt.annotate(f'Best: {best_queue_algo} (Avg: {queue_avgs[best_queue_algo]:.2f})',
             xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
             bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))
plt.savefig("comparison_queue_length.png")
plt.close()
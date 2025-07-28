import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Current working directory:", os.getcwd())

# Load data function with graceful fallback
def load_data(prefix):
    try:
        episode_history = np.load(f"{prefix}_episode_history.npy")
        reward_history = np.load(f"{prefix}_reward_history.npy")
        queue_history = np.load(f"{prefix}_queue_history.npy")
        eval_episode_history = np.load(f"{prefix}_eval_episode_history.npy")
        eval_reward_history = np.load(f"{prefix}_eval_reward_history.npy")
        eval_queue_history = np.load(f"{prefix}_eval_queue_history.npy")
        return (episode_history, reward_history, queue_history,
                eval_episode_history, eval_reward_history, eval_queue_history)
    except FileNotFoundError as e:
        print(f"Warning: Missing data files for {prefix}: {e}")
        # Return empty arrays for eval if missing, raise if primary data missing
        try:
            episode_history = np.load(f"{prefix}_episode_history.npy")
            reward_history = np.load(f"{prefix}_reward_history.npy")
            queue_history = np.load(f"{prefix}_queue_history.npy")
        except FileNotFoundError as e2:
            print(f"Error: Missing primary training data files for {prefix}: {e2}")
            raise e2
        return (episode_history, reward_history, queue_history,
                np.array([]), np.array([]), np.array([]))


# Load all data
ql_data = load_data("ql")
dqn_data = load_data("dqn")
ppo_data = load_data("ppo")
ft_data = load_data("ft")  # Fixed-Time may have no eval data; handled gracefully

# Unpack loaded data for clarity
(ql_ep, ql_r, ql_q, ql_eval_ep, ql_eval_r, ql_eval_q) = ql_data
(dqn_ep, dqn_r, dqn_q, dqn_eval_ep, dqn_eval_r, dqn_eval_q) = dqn_data
(ppo_ep, ppo_r, ppo_q, ppo_eval_ep, ppo_eval_r, ppo_eval_q) = ppo_data
(ft_ep, ft_r, ft_q, ft_eval_ep, ft_eval_r, ft_eval_q) = ft_data

# Align lengths of training episodes (truncate to shortest)
max_episodes = min(len(ql_ep), len(dqn_ep), len(ppo_ep), len(ft_ep))
ql_ep, ql_r, ql_q = ql_ep[:max_episodes], ql_r[:max_episodes], ql_q[:max_episodes]
dqn_ep, dqn_r, dqn_q = dqn_ep[:max_episodes], dqn_r[:max_episodes], dqn_q[:max_episodes]
ppo_ep, ppo_r, ppo_q = ppo_ep[:max_episodes], ppo_r[:max_episodes], ppo_q[:max_episodes]
ft_ep, ft_r, ft_q = ft_ep[:max_episodes], ft_r[:max_episodes], ft_q[:max_episodes]

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

# --- 1. Training Plots ---

# Cumulative Reward Training Plot
plt.figure(figsize=(14, 8))
plt.plot(ql_ep, ql_r, marker=markers["Q-learning"], linestyle='-', label="Q-learning Training Reward", color=colors["Q-learning"])
plt.plot(dqn_ep, dqn_r, marker=markers["DQN"], linestyle='-', label="DQN Training Reward", color=colors["DQN"])
plt.plot(ppo_ep, ppo_r, marker=markers["PPO"], linestyle='-', label="PPO Training Reward", color=colors["PPO"])
plt.plot(ft_ep, ft_r, marker=markers["Fixed-Time"], linestyle='--', label="Fixed-Time Training Reward", color=colors["Fixed-Time"])
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Training: Cumulative Reward over Episodes")
plt.legend()
plt.grid(True)
plt.savefig("training_cumulative_reward.png")
plt.close()

# Average Queue Length Training Plot
plt.figure(figsize=(14, 8))
plt.plot(ql_ep, ql_q, marker=markers["Q-learning"], linestyle='-', label="Q-learning Training Avg Queue", color=colors["Q-learning"])
plt.plot(dqn_ep, dqn_q, marker=markers["DQN"], linestyle='-', label="DQN Training Avg Queue", color=colors["DQN"])
plt.plot(ppo_ep, ppo_q, marker=markers["PPO"], linestyle='-', label="PPO Training Avg Queue", color=colors["PPO"])
plt.plot(ft_ep, ft_q, marker=markers["Fixed-Time"], linestyle='--', label="Fixed-Time Training Avg Queue", color=colors["Fixed-Time"])
plt.xlabel("Episode")
plt.ylabel("Average Queue Length")
plt.title("Training: Average Queue Length over Episodes")
plt.legend()
plt.grid(True)
plt.savefig("training_average_queue_length.png")
plt.close()


# --- 2. Evaluation Plots (only for algorithms with eval data) ---

def plot_eval_metric(eval_eps, eval_vals, label, ylabel, title, filename, color, marker, linewidth=2):
    if len(eval_eps) == 0 or len(eval_vals) == 0:
        return
    plt.plot(eval_eps, eval_vals, marker=marker, linestyle='-', label=label, color=color, linewidth=linewidth)

plt.figure(figsize=(14, 8))
plot_eval_metric(ql_eval_ep, ql_eval_r, "Q-learning Eval Reward", "Evaluation Cumulative Reward", "Evaluation: Cumulative Reward over Episodes", None, colors["Q-learning"], markers["Q-learning"])
plot_eval_metric(dqn_eval_ep, dqn_eval_r, "DQN Eval Reward", None, None, None, colors["DQN"], markers["DQN"])
plot_eval_metric(ppo_eval_ep, ppo_eval_r, "PPO Eval Reward", None, None, None, colors["PPO"], markers["PPO"])
plt.xlabel("Episode")
plt.ylabel("Evaluation Cumulative Reward")
plt.title("Evaluation: Cumulative Reward over Episodes")
plt.legend()
plt.grid(True)
plt.savefig("evaluation_cumulative_reward.png")
plt.close()

plt.figure(figsize=(14, 8))
plot_eval_metric(ql_eval_ep, ql_eval_q, "Q-learning Eval Queue", "Evaluation Average Queue Length", "Evaluation: Average Queue Length over Episodes", None, colors["Q-learning"], markers["Q-learning"])
plot_eval_metric(dqn_eval_ep, dqn_eval_q, "DQN Eval Queue", None, None, None, colors["DQN"], markers["DQN"])
plot_eval_metric(ppo_eval_ep, ppo_eval_q, "PPO Eval Queue", None, None, None, colors["PPO"], markers["PPO"])
plt.xlabel("Episode")
plt.ylabel("Evaluation Average Queue Length")
plt.title("Evaluation: Average Queue Length over Episodes")
plt.legend()
plt.grid(True)
plt.savefig("evaluation_average_queue_length.png")
plt.close()


# --- 3. Determine best models based on evaluation averages ---

# Define function to get mean evaluation metric safely
def safe_mean(arr):
    return np.mean(arr) if len(arr) > 0 else float('-inf')

eval_reward_avgs = {
    "Q-learning": safe_mean(ql_eval_r),
    "DQN": safe_mean(dqn_eval_r),
    "PPO": safe_mean(ppo_eval_r),
    # Fixed-Time has no evaluation data; use training reward average as fallback or skip
    "Fixed-Time": np.mean(ft_r)  # or -inf if you want to exclude FT from eval comparison
}

eval_queue_avgs = {
    "Q-learning": np.mean(ql_eval_q) if len(ql_eval_q) > 0 else float('inf'),
    "DQN": np.mean(dqn_eval_q) if len(dqn_eval_q) > 0 else float('inf'),
    "PPO": np.mean(ppo_eval_q) if len(ppo_eval_q) > 0 else float('inf'),
    "Fixed-Time": np.mean(ft_q)
}

best_eval_reward_algo = max(eval_reward_avgs, key=eval_reward_avgs.get)
best_eval_queue_algo = min(eval_queue_avgs, key=eval_queue_avgs.get)

print(f"Best algorithm based on evaluation cumulative reward: {best_eval_reward_algo} (Avg: {eval_reward_avgs[best_eval_reward_algo]:.2f})")
print(f"Best algorithm based on evaluation average queue length: {best_eval_queue_algo} (Avg: {eval_queue_avgs[best_eval_queue_algo]:.2f})")


# --- 4. Save best models summary to text file ---

with open("best_models_summary.txt", "w") as f:
    f.write("Best Models Summary (based on Evaluation Metrics):\n\n")
    f.write(f"Q-learning: Avg Eval Reward = {eval_reward_avgs['Q-learning']:.2f}, Avg Eval Queue Length = {eval_queue_avgs['Q-learning']:.2f}\n")
    f.write(f"DQN:       Avg Eval Reward = {eval_reward_avgs['DQN']:.2f}, Avg Eval Queue Length = {eval_queue_avgs['DQN']:.2f}\n")
    f.write(f"PPO:       Avg Eval Reward = {eval_reward_avgs['PPO']:.2f}, Avg Eval Queue Length = {eval_queue_avgs['PPO']:.2f}\n")
    f.write(f"Fixed-Time: Avg Training Reward (no evaluation) = {np.mean(ft_r):.2f}, Avg Training Queue Length = {np.mean(ft_q):.2f}\n\n")

    f.write(f"Best algorithm for cumulative reward (evaluation): {best_eval_reward_algo} (Avg: {eval_reward_avgs[best_eval_reward_algo]:.2f})\n")
    f.write(f"Best algorithm for average queue length (evaluation): {best_eval_queue_algo} (Avg: {eval_queue_avgs[best_eval_queue_algo]:.2f})\n")

print("Best models summary saved to best_models_summary.txt")
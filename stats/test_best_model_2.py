import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ===== Algorithms to analyze =====
algorithms = [
    {"name": "Q-learning", "short": "ql", "color": '#9467bd', "marker": 'd',  "train_style": '-', "eval_style": '--'},
    {"name": "DQN",        "short": "dqn", "color": '#1f77b4', "marker": 'o', "train_style": '-', "eval_style": '--'},
    {"name": "PPO",        "short": "ppo", "color": '#2ca02c', "marker": '^', "train_style": '-', "eval_style": '--'},
    {"name": "Fixed-Time", "short": "ft",  "color": '#ff7f0e', "marker": 's', "train_style": '--', "eval_style": ':'}
]

# ===== Load histories ======
histories = {}
for algo in algorithms:
    s = algo["short"]
    histories[s] = {}
    # Training
    try:
        histories[s]["episode"] = np.load(f"{s}_episode_history.npy")
        histories[s]["reward"] = np.load(f"{s}_reward_history.npy")
        histories[s]["queue"]  = np.load(f"{s}_queue_history.npy")
    except Exception as e:
        print(f"Warning: Can't load training for {algo['name']}: {e}")
    # Evaluation
    try:
        histories[s]["eval_episode"] = np.load(f"{s}_eval_episode_history.npy")
        histories[s]["eval_reward"]  = np.load(f"{s}_eval_reward_history.npy")
        histories[s]["eval_queue"]   = np.load(f"{s}_eval_queue_history.npy")
    except Exception:
        histories[s]["eval_episode"] = histories[s]["eval_reward"] = histories[s]["eval_queue"] = None

# ====== Harmonize training episode length for fair comparison =====
min_episodes = min(
    [len(histories[s]["episode"]) for s in histories if "episode" in histories[s] and histories[s]["episode"] is not None]
)
for s in histories:
    for k in ["episode", "reward", "queue"]:
        arr = histories[s].get(k)
        if arr is not None and len(arr) > min_episodes:
            histories[s][k] = arr[:min_episodes]

# ======= Best/worst finders =======
def find_best_worst(histories, metric, higher_is_better=True):
    avgs = {}
    for algo in algorithms:
        s = algo["short"]
        dat = histories[s].get(metric)
        avgs[algo["name"]] = np.mean(dat) if dat is not None else None
    valid = {k: v for k, v in avgs.items() if v is not None}
    if not valid:
        return None, None, avgs
    if higher_is_better:
        best = max(valid, key=valid.get)
        worst = min(valid, key=valid.get)
    else:
        best = min(valid, key=valid.get)
        worst = max(valid, key=valid.get)
    return best, worst, avgs

# ======= Plotting functions =======
def plot_training_curves(metric, ylabel, filename, title, higher_is_better=True):
    best, worst, avgs = find_best_worst(histories, metric, higher_is_better)
    plt.figure(figsize=(14, 8))
    for algo in algorithms:
        s = algo["short"]
        lw = 3 if algo["name"] == best else (3 if algo["name"] == worst else 1)
        ls = '-' if algo["name"] != worst else 'dashdot'
        arr_ep = histories[s].get("episode")
        arr_m  = histories[s].get(metric)
        if arr_ep is not None and arr_m is not None:
            plt.plot(
                arr_ep, arr_m,
                marker=algo["marker"], linestyle=ls, color=algo["color"],
                label=f"{algo['name']} (Avg: {avgs[algo['name']]:.1f})",
                linewidth=lw
            )
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    if best: plt.annotate(f'Best: {best} ({avgs[best]:.2f})', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))
    if worst: plt.annotate(f'Worst: {worst} ({avgs[worst]:.2f})', xy=(0.05, 0.89), xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))
    plt.savefig(filename)
    plt.close()

def plot_evaluation_curves(metric, ylabel, filename, title, higher_is_better=True):
    best, worst, avgs = find_best_worst({s: {"eval": histories[s][f"eval_{metric}"]} for s in histories},
                                        "eval", higher_is_better)
    plt.figure(figsize=(14, 8))
    for algo in algorithms:
        s = algo["short"]
        arr_ep = histories[s].get("eval_episode")
        arr_m  = histories[s].get(f"eval_{metric}")
        lw = 3 if algo["name"] == best else (3 if algo["name"] == worst else 1)
        ls = '-' if algo["name"] != worst else 'dashdot'
        if arr_ep is not None and arr_m is not None:
            plt.plot(
                arr_ep, arr_m,
                marker=algo["marker"], linestyle=ls, color=algo["color"],
                label=f"{algo['name']} (Eval, Avg: {np.mean(arr_m):.1f})",
                linewidth=lw
            )
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    if best: plt.annotate(f'Best: {best}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))
    if worst: plt.annotate(f'Worst: {worst}', xy=(0.05, 0.89), xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))
    plt.savefig(filename)
    plt.close()

def plot_combined_curves(metric, ylabel, filename, title, higher_is_better=True):
    best, worst, avgs = find_best_worst(histories, metric, higher_is_better)
    plt.figure(figsize=(14, 8))
    for algo in algorithms:
        s = algo["short"]
        # Training
        lw = 3 if algo["name"] == best else (3 if algo["name"] == worst else 1)
        ls = '-' if algo["name"] != worst else 'dashdot'
        arr_ep = histories[s].get("episode")
        arr_m  = histories[s].get(metric)
        if arr_ep is not None and arr_m is not None:
            plt.plot(
                arr_ep, arr_m,
                marker=algo["marker"], linestyle=ls, color=algo["color"], alpha=0.7,
                label=f"{algo['name']} (Train, Avg: {avgs[algo['name']]:.1f})",
                linewidth=lw
            )
        # Evaluation
        arr_eval_ep = histories[s].get("eval_episode")
        arr_eval_m  = histories[s].get(f"eval_{metric}")
        if arr_eval_ep is not None and arr_eval_m is not None:
            plt.plot(
                arr_eval_ep, arr_eval_m,
                marker=algo["marker"], linestyle=algo["eval_style"], color=algo["color"],
                linewidth=2, label=f"{algo['name']} (Eval)"
            )
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    if best: plt.annotate(f'Best: {best} ({avgs[best]:.2f})', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))
    if worst: plt.annotate(f'Worst: {worst} ({avgs[worst]:.2f})', xy=(0.05, 0.89), xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))
    plt.savefig(filename)
    plt.close()

# ========== Generate plots ===========
plot_training_curves("reward", "Cumulative Reward", "comparison_training_cumulative_reward.png", "Training Cumulative Reward per Episode", higher_is_better=True)
plot_training_curves("queue",  "Average Queue Length", "comparison_training_queue_length.png", "Training Average Queue Length per Episode", higher_is_better=False)
plot_evaluation_curves("reward", "Cumulative Reward", "comparison_eval_cumulative_reward.png", "Evaluation Cumulative Reward per Evaluation", higher_is_better=True)
plot_evaluation_curves("queue",  "Average Queue Length", "comparison_eval_queue_length.png", "Evaluation Average Queue Length per Evaluation", higher_is_better=False)
plot_combined_curves("reward", "Cumulative Reward", "comparison_combined_cumulative_reward.png", "Training/Evaluation Cumulative Reward per Episode", higher_is_better=True)
plot_combined_curves("queue",  "Average Queue Length", "comparison_combined_queue_length.png", "Training/Evaluation Avg Queue Length per Episode", higher_is_better=False)

print("All plots generated (best: bold, worst: bold+dashed)!")
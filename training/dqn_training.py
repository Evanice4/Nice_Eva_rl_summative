"""
DQN Training for SokoPrice
Runs 10 hyperparameter experiments and saves results and best model.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import SokoPriceEnv

MODELS_DIR = "models/dqn"
LOGS_DIR   = "logs/dqn"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)


#  10 Hyperparameter configurations
DQN_EXPERIMENTS = [
    # lr,      gamma, bs,  buf,    eps_start, eps_end, eps_frac, tau,   net_arch,        label
    (1e-3,  0.99, 64,  50_000, 1.0, 0.05, 0.2,  1.0,   [64,  64],       "E01_baseline"),
    (5e-4,  0.99, 64,  50_000, 1.0, 0.05, 0.2,  1.0,   [64,  64],       "E02_low_lr"),
    (1e-3,  0.95, 64,  50_000, 1.0, 0.05, 0.2,  1.0,   [64,  64],       "E03_low_gamma"),
    (1e-3,  0.99, 128, 50_000, 1.0, 0.05, 0.2,  1.0,   [64,  64],       "E04_large_batch"),
    (1e-3,  0.99, 64,  100_000,1.0, 0.05, 0.3,  1.0,   [64,  64],       "E05_large_buffer"),
    (1e-3,  0.99, 64,  50_000, 1.0, 0.01, 0.5,  1.0,   [64,  64],       "E06_low_eps_end"),
    (1e-3,  0.99, 64,  50_000, 1.0, 0.05, 0.2,  0.005, [64,  64],       "E07_soft_update"),
    (1e-3,  0.99, 64,  50_000, 1.0, 0.05, 0.2,  1.0,   [128, 128],      "E08_wide_net"),
    (1e-3,  0.99, 64,  50_000, 1.0, 0.05, 0.2,  1.0,   [64,  64,  64],  "E09_deep_net"),
    (3e-4,  0.995,64,  80_000, 1.0, 0.02, 0.4,  0.005, [128, 128, 64],  "E10_best_guess"),
]

TOTAL_TIMESTEPS = 80_000   


class RewardLogger(BaseCallback):
    """Logs episode rewards during training."""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current = 0.0

    def _on_step(self):
        reward = self.locals["rewards"][0]
        self._current += reward
        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(self._current)
            self._current = 0.0
        return True


def train_dqn(exp_cfg, exp_id, total_timesteps=TOTAL_TIMESTEPS):
    lr, gamma, bs, buf, eps0, eps1, eps_f, tau, arch, label = exp_cfg

    env      = Monitor(SokoPriceEnv())
    eval_env = Monitor(SokoPriceEnv())

    callback = RewardLogger()

    model = DQN(
        policy             = "MlpPolicy",
        env                = env,
        learning_rate      = lr,
        gamma              = gamma,
        batch_size         = bs,
        buffer_size        = buf,
        exploration_initial_eps = eps0,
        exploration_final_eps   = eps1,
        exploration_fraction    = eps_f,
        tau                = tau,
        policy_kwargs      = dict(net_arch=arch),
        verbose            = 0,
        tensorboard_log    = LOGS_DIR,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)

    save_path = os.path.join(MODELS_DIR, label)
    model.save(save_path)
    print(f"  [{exp_id:02d}/10] {label:<30} | "
          f"Mean reward: {np.mean(callback.episode_rewards[-10:]):.2f}")

    return {
        "experiment":    label,
        "learning_rate": lr,
        "gamma":         gamma,
        "batch_size":    bs,
        "buffer_size":   buf,
        "eps_start":     eps0,
        "eps_end":       eps1,
        "eps_fraction":  eps_f,
        "tau":           tau,
        "net_arch":      str(arch),
        "mean_reward_last10": round(float(np.mean(callback.episode_rewards[-10:])), 3),
        "mean_reward_all":    round(float(np.mean(callback.episode_rewards))       , 3),
        "episode_rewards": callback.episode_rewards,
    }


def run_all_dqn_experiments():
    print("\n" + "═" * 60)
    print("  DQN Hyperparameter Experiments - SokoPrice")
    print("═" * 60)
    results = []
    for i, cfg in enumerate(DQN_EXPERIMENTS):
        res = train_dqn(cfg, i + 1)
        results.append(res)

    # Save table 
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "episode_rewards"}
                       for r in results])
    df.to_csv(os.path.join(LOGS_DIR, "dqn_results.csv"), index=False)
    print(f"\n Results table -> {LOGS_DIR}/dqn_results.csv")
    print(df[["experiment", "learning_rate", "gamma", "batch_size",
              "eps_end", "tau", "net_arch", "mean_reward_last10"]].to_string(index=False))

    # Plot reward curves 
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), facecolor="#0F1423")
    fig.suptitle("DQN - Cumulative Episode Rewards per Experiment",
                 color="white", fontsize=14)
    for ax, res in zip(axes.flat, results):
        r  = res["episode_rewards"]
        ep = range(len(r))
        ax.plot(ep, r, color="#32C8FF", linewidth=1.2, alpha=0.7)
        if len(r) > 5:
            smooth = pd.Series(r).rolling(5).mean()
            ax.plot(ep, smooth, color="#FFD232", linewidth=2)
        ax.set_facecolor("#16203A")
        ax.set_title(res["experiment"], color="white", fontsize=8)
        ax.set_xlabel("Episode", color="#8C9BB9", fontsize=7)
        ax.set_ylabel("Reward",  color="#8C9BB9", fontsize=7)
        ax.tick_params(colors="#8C9BB9", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2D3D60")
    plt.tight_layout()
    plt.savefig(os.path.join(LOGS_DIR, "dqn_reward_curves.png"), dpi=150, bbox_inches="tight")
    print(f"Reward curves -> {LOGS_DIR}/dqn_reward_curves.png")

    # Plot objective (TD loss proxy) via mean reward trend
    fig2, ax2 = plt.subplots(figsize=(12, 5), facecolor="#0F1423")
    ax2.set_facecolor("#16203A")
    ax2.set_title("DQN - Best Experiment Cumulative Reward", color="white")
    best = max(results, key=lambda x: x["mean_reward_last10"])
    r    = best["episode_rewards"]
    ax2.plot(r, color="#32C8FF", alpha=0.5, linewidth=1, label="Episode reward")
    if len(r) > 10:
        ax2.plot(pd.Series(r).rolling(10).mean(), color="#FFD232", linewidth=2, label="Rolling mean")
    ax2.legend(facecolor="#16203A", labelcolor="white")
    ax2.set_xlabel("Episode", color="#8C9BB9")
    ax2.set_ylabel("Reward",  color="#8C9BB9")
    ax2.tick_params(colors="#8C9BB9")
    plt.savefig(os.path.join(LOGS_DIR, "dqn_best_curve.png"), dpi=150, bbox_inches="tight")
    print(f" Best curve      -> {LOGS_DIR}/dqn_best_curve.png")

    best_name = best["experiment"]
    print(f"\n Best DQN config: {best_name}  "
          f"(mean last-10 reward: {best['mean_reward_last10']:.2f})")
    return results, best_name


if __name__ == "__main__":
    run_all_dqn_experiments()
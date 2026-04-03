"""
Policy Gradient Training for SokoPrice
Implements REINFORCE (custom) and PPO (Stable Baselines3).
Runs 10 hyperparameter experiments each and saves results.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import SokoPriceEnv

MODELS_DIR_PPO       = "models/pg/ppo"
MODELS_DIR_REINFORCE = "models/pg/reinforce"
LOGS_DIR             = "logs/pg"
for d in [MODELS_DIR_PPO, MODELS_DIR_REINFORCE, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

TOTAL_TIMESTEPS_PPO       = 80_000
REINFORCE_EPISODES        = 300


#  1. REINFORCE (custom PyTorch implementation)

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def reinforce_train(lr, gamma, hidden, n_episodes, label, entropy_coef=0.01):
    env     = SokoPriceEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy  = PolicyNet(obs_dim, act_dim, hidden)
    opt     = optim.Adam(policy.parameters(), lr=lr)

    ep_rewards = []
    entropy_log = []

    for ep in range(n_episodes):
        obs, _  = env.reset()
        states, actions, rewards_ep = [], [], []
        done = False

        while not done:
            x     = torch.FloatTensor(obs).unsqueeze(0)
            probs = policy(x)
            dist  = torch.distributions.Categorical(probs)
            a     = dist.sample().item()
            obs, r, terminated, truncated, _ = env.step(a)
            done  = terminated or truncated
            states.append(x)
            actions.append(a)
            rewards_ep.append(r)

        # Discounted returns
        G, returns = 0.0, []
        for r in reversed(rewards_ep):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        loss = torch.tensor(0.0)
        ep_entropy = 0.0
        for x, a, G_t in zip(states, actions, returns):
            probs   = policy(x)
            dist    = torch.distributions.Categorical(probs)
            log_p   = dist.log_prob(torch.tensor(a))
            entropy = dist.entropy()
            loss    = loss - (log_p * G_t + entropy_coef * entropy)
            ep_entropy += entropy.item()

        opt.zero_grad()
        loss.backward()
        opt.step()

        ep_rewards.append(sum(rewards_ep))
        entropy_log.append(ep_entropy / max(len(rewards_ep), 1))

        if (ep + 1) % 50 == 0:
            print(f"    {label} | Ep {ep+1:>3}/{n_episodes} | "
                  f"Mean reward (last 20): {np.mean(ep_rewards[-20:]):.2f}")

    torch.save(policy.state_dict(), os.path.join(MODELS_DIR_REINFORCE, f"{label}.pt"))
    env.close()
    return ep_rewards, entropy_log


REINFORCE_EXPERIMENTS = [
    # lr,    gamma, hidden,         entropy_coef, label
    (1e-3,  0.99,  [64, 64],        0.01,  "R01_baseline"),
    (5e-4,  0.99,  [64, 64],        0.01,  "R02_low_lr"),
    (2e-3,  0.99,  [64, 64],        0.01,  "R03_high_lr"),
    (1e-3,  0.95,  [64, 64],        0.01,  "R04_low_gamma"),
    (1e-3,  0.99,  [128, 128],      0.01,  "R05_wide_net"),
    (1e-3,  0.99,  [64, 64, 64],    0.01,  "R06_deep_net"),
    (1e-3,  0.99,  [64, 64],        0.05,  "R07_high_entropy"),
    (1e-3,  0.99,  [64, 64],        0.001, "R08_low_entropy"),
    (1e-3,  0.995, [128, 64],       0.02,  "R09_high_gamma"),
    (3e-4,  0.995, [128, 128, 64],  0.02,  "R10_best_guess"),
]


def run_all_reinforce():
    print("\n" + "═" * 60)
    print("  REINFORCE Hyperparameter Experiments - SokoPrice")
    print("═" * 60)
    all_results = []
    all_ep_rewards = []
    all_entropies  = []

    for i, (lr, gamma, hidden, ent, label) in enumerate(REINFORCE_EXPERIMENTS):
        print(f"\n  [{i+1:02d}/10] {label}")
        ep_rw, ent_log = reinforce_train(lr, gamma, hidden, REINFORCE_EPISODES, label, ent)
        all_ep_rewards.append(ep_rw)
        all_entropies.append(ent_log)
        all_results.append({
            "experiment":    label,
            "learning_rate": lr,
            "gamma":         gamma,
            "hidden":        str(hidden),
            "entropy_coef":  ent,
            "episodes":      REINFORCE_EPISODES,
            "mean_reward_last20": round(float(np.mean(ep_rw[-20:])), 3),
            "mean_reward_all":    round(float(np.mean(ep_rw)),       3),
        })

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(LOGS_DIR, "reinforce_results.csv"), index=False)
    print(f"\n REINFORCE table -> {LOGS_DIR}/reinforce_results.csv")
    print(df[["experiment", "learning_rate", "gamma", "entropy_coef",
              "hidden", "mean_reward_last20"]].to_string(index=False))

    # Plot reward curves
    _plot_pg_curves(all_ep_rewards, REINFORCE_EXPERIMENTS, "REINFORCE",
                    os.path.join(LOGS_DIR, "reinforce_reward_curves.png"), x_label="Episode")
    # Entropy curves 
    _plot_entropy_curves(all_entropies, REINFORCE_EXPERIMENTS,
                         os.path.join(LOGS_DIR, "reinforce_entropy_curves.png"))

    best = max(all_results, key=lambda x: x["mean_reward_last20"])
    print(f"\n Best REINFORCE: {best['experiment']}  "
          f"(mean last-20: {best['mean_reward_last20']:.2f})")
    return all_results


#  2. PPO (Stable Baselines3)

PPO_EXPERIMENTS = [
    # lr,    gamma, n_steps, batch_size, n_epochs, ent_coef, clip_range, gae_lam, net_arch,    label
    (3e-4,  0.99,  2048,  64,  10, 0.01, 0.2, 0.95, [64,  64],       "P01_baseline"),
    (1e-4,  0.99,  2048,  64,  10, 0.01, 0.2, 0.95, [64,  64],       "P02_low_lr"),
    (1e-3,  0.99,  2048,  64,  10, 0.01, 0.2, 0.95, [64,  64],       "P03_high_lr"),
    (3e-4,  0.95,  2048,  64,  10, 0.01, 0.2, 0.95, [64,  64],       "P04_low_gamma"),
    (3e-4,  0.99,  1024,  32,  10, 0.01, 0.2, 0.95, [64,  64],       "P05_short_rollout"),
    (3e-4,  0.99,  2048,  128, 10, 0.01, 0.2, 0.95, [64,  64],       "P06_large_batch"),
    (3e-4,  0.99,  2048,  64,  20, 0.01, 0.2, 0.95, [64,  64],       "P07_more_epochs"),
    (3e-4,  0.99,  2048,  64,  10, 0.05, 0.2, 0.95, [64,  64],       "P08_high_entropy"),
    (3e-4,  0.99,  2048,  64,  10, 0.01, 0.3, 0.95, [128, 128],      "P09_wide_clip"),
    (2e-4,  0.995, 2048,  64,  15, 0.02, 0.2, 0.98, [128, 128, 64], "P10_best_guess"),
]


class PPORewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current = 0.0

    def _on_step(self):
        self._current += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current)
            self._current = 0.0
        return True


def run_all_ppo():
    print("\n" + "═" * 60)
    print("  PPO Hyperparameter Experiments - SokoPrice")
    print("═" * 60)
    all_results    = []
    all_ep_rewards = []

    for i, (lr, gamma, n_steps, bs, n_ep, ent, clip, lam, arch, label) in enumerate(PPO_EXPERIMENTS):
        print(f"\n  [{i+1:02d}/10] {label}")
        env      = Monitor(SokoPriceEnv())
        callback = PPORewardLogger()

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate     = lr,
            gamma             = gamma,
            n_steps           = n_steps,
            batch_size        = bs,
            n_epochs          = n_ep,
            ent_coef          = ent,
            clip_range        = clip,
            gae_lambda        = lam,
            policy_kwargs     = dict(net_arch=arch),
            verbose           = 0,
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS_PPO, callback=callback)
        model.save(os.path.join(MODELS_DIR_PPO, label))

        ep_rw = callback.episode_rewards
        all_ep_rewards.append(ep_rw)
        all_results.append({
            "experiment":    label,
            "learning_rate": lr,
            "gamma":         gamma,
            "n_steps":       n_steps,
            "batch_size":    bs,
            "n_epochs":      n_ep,
            "ent_coef":      ent,
            "clip_range":    clip,
            "gae_lambda":    lam,
            "net_arch":      str(arch),
            "mean_reward_last10": round(float(np.mean(ep_rw[-10:]) if ep_rw else 0), 3),
            "mean_reward_all":    round(float(np.mean(ep_rw)       if ep_rw else 0), 3),
        })
        print(f"    Mean last-10 reward: {all_results[-1]['mean_reward_last10']:.2f}")
        env.close()

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(LOGS_DIR, "ppo_results.csv"), index=False)
    print(f"\n PPO table → {LOGS_DIR}/ppo_results.csv")
    print(df[["experiment", "learning_rate", "gamma", "n_steps", "batch_size",
              "ent_coef", "clip_range", "mean_reward_last10"]].to_string(index=False))

    _plot_pg_curves(all_ep_rewards, PPO_EXPERIMENTS, "PPO",
                    os.path.join(LOGS_DIR, "ppo_reward_curves.png"), x_label="Episode")

    best = max(all_results, key=lambda x: x["mean_reward_last10"])
    print(f"\nBest PPO: {best['experiment']}  "
          f"(mean last-10: {best['mean_reward_last10']:.2f})")
    return all_results

#  Shared plotting utilities

def _plot_pg_curves(all_ep_rewards, experiments, algo_name, save_path, x_label="Episode"):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), facecolor="#0F1423")
    fig.suptitle(f"{algo_name} - Episode Rewards per Experiment", color="white", fontsize=14)
    for ax, ep_rw, exp in zip(axes.flat, all_ep_rewards, experiments):
        label = exp[-1] if isinstance(exp[-1], str) else str(exp[-1])
        ax.plot(ep_rw, color="#A0E0FF", linewidth=1, alpha=0.6)
        if len(ep_rw) > 5:
            ax.plot(pd.Series(ep_rw).rolling(5).mean(), color="#FFD232", linewidth=2)
        ax.set_facecolor("#16203A")
        ax.set_title(label, color="white", fontsize=8)
        ax.set_xlabel(x_label, color="#8C9BB9", fontsize=7)
        ax.set_ylabel("Reward",  color="#8C9BB9", fontsize=7)
        ax.tick_params(colors="#8C9BB9", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2D3D60")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"{algo_name} curves -> {save_path}")


def _plot_entropy_curves(all_entropies, experiments, save_path):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), facecolor="#0F1423")
    fig.suptitle("REINFORCE - Policy Entropy per Experiment", color="white", fontsize=14)
    for ax, ent, exp in zip(axes.flat, all_entropies, experiments):
        label = exp[-1] if isinstance(exp[-1], str) else str(exp[-1])
        ax.plot(ent, color="#32C8A0", linewidth=1.2)
        ax.set_facecolor("#16203A")
        ax.set_title(label, color="white", fontsize=8)
        ax.set_xlabel("Episode", color="#8C9BB9", fontsize=7)
        ax.set_ylabel("Entropy",  color="#8C9BB9", fontsize=7)
        ax.tick_params(colors="#8C9BB9", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2D3D60")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Entropy curves -> {save_path}")


def plot_comparison(dqn_results, reinforce_results, ppo_results):
    """Compare best of each algorithm side by side."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0F1423")
    ax.set_facecolor("#16203A")
    ax.set_title("Algorithm Comparison - Best Config Mean Rewards", color="white", fontsize=14)

    methods = [
        ("DQN",       dqn_results,       "#32C8FF"),
        ("REINFORCE", reinforce_results,  "#A0E040"),
        ("PPO",       ppo_results,        "#FFB030"),
    ]
    key_last = {"DQN": "mean_reward_last10", "REINFORCE": "mean_reward_last20", "PPO": "mean_reward_last10"}

    xs, ys, cs = [], [], []
    for name, results, col in methods:
        best = max(results, key=lambda x: x.get(key_last[name], 0))
        xs.append(name)
        ys.append(best.get(key_last[name], 0))
        cs.append(col)
        ax.bar(name, best.get(key_last[name], 0), color=col, alpha=0.85, width=0.5)
        ax.text(name, best.get(key_last[name], 0) + 0.2, f"{best.get(key_last[name], 0):.2f}",
                ha="center", color="white", fontsize=11)

    ax.set_ylabel("Mean Reward (best config)", color="#8C9BB9")
    ax.tick_params(colors="#8C9BB9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2D3D60")
    plt.savefig(os.path.join(LOGS_DIR, "algorithm_comparison.png"), dpi=150, bbox_inches="tight")
    print(f"Comparison chart -> {LOGS_DIR}/algorithm_comparison.png")


if __name__ == "__main__":
    reinforce_results = run_all_reinforce()
    ppo_results       = run_all_ppo()
    print("\nDone. Load DQN results from logs/dqn/dqn_results.csv for comparison plotting.")
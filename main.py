"""
SokoPrice RL Entry Point
Loads the best-performing model and runs a demonstration
with full pygame visualisation saved as a GIF.

Usage:
    python main.py                    # auto-detect best model
    python main.py --algo dqn
    python main.py --algo ppo
    python main.py --algo reinforce
    python main.py --random           # random-agent demo only
"""

import os
import sys
import argparse
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment.custom_env import SokoPriceEnv, FOOD_ITEMS, BASE_PRICES, MAX_DAYS
from environment.rendering import PILRenderer, run_random_agent_demo_3d, _action_name

def run_random_agent_demo(save_path="random_agent_demo.gif", headless=True, n_steps=MAX_DAYS):
    return run_random_agent_demo_3d(save_path, n_steps)


#  Model loaders
def load_sb3_model(algo, model_path):
    if algo == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(model_path)
    elif algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    return None


def load_reinforce_model(model_path, obs_dim=25, act_dim=27):
    from training.pg_training import PolicyNet
    import re
    # parse hidden size from filename heuristic 
    hidden = [128, 128, 64] if "best" in model_path.lower() else [64, 64]
    net = PolicyNet(obs_dim, act_dim, hidden)
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()
    return net

#  Best model discovery
def find_best_model():
    """Read CSV results and return (algo, path) of best model."""
    import pandas as pd
    best_score = -1e9
    best_algo  = None
    best_path  = None

    checks = [
        ("dqn",       "logs/dqn/dqn_results.csv",         "mean_reward_last10", "models/dqn"),
        ("ppo",       "logs/pg/ppo_results.csv",           "mean_reward_last10", "models/pg/ppo"),
        ("reinforce", "logs/pg/reinforce_results.csv",     "mean_reward_last20", "models/pg/reinforce"),
    ]
    for algo, csv_path, col, model_dir in checks:
        if not os.path.exists(csv_path):
            continue
        df    = pd.read_csv(csv_path)
        idx   = df[col].idxmax()
        score = df.loc[idx, col]
        name  = df.loc[idx, "experiment"]
        if algo == "reinforce":
            path = os.path.join(model_dir, f"{name}.pt")
        else:
            path = os.path.join(model_dir, name)
        if score > best_score and os.path.exists(path if algo == "reinforce" else path + ".zip"):
            best_score = score
            best_algo  = algo
            best_path  = path
            print(f"  [{algo.upper():>9}] {name:<30} score={score:.2f}")

    return best_algo, best_path

#  Run demonstration
def run_demo(algo, model_path, save_path="best_agent_demo.gif", headless=True):
    print(f"\n{'═'*55}")
    print(f"  SokoPrice — {algo.upper()} Agent Demo")
    print(f"  Model: {model_path}")
    print(f"{'═'*55}")

    env      = SokoPriceEnv()
    renderer = PILRenderer()

    # Load model
    if algo in ("dqn", "ppo"):
        model = load_sb3_model(algo, model_path)
        def predict(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)
    else:
        model = load_reinforce_model(model_path)
        def predict(obs):
            with torch.no_grad():
                x     = torch.FloatTensor(obs).unsqueeze(0)
                probs = model(x)
                return int(probs.argmax(dim=1).item())

    obs, info = env.reset()
    total_reward = 0.0
    frames = []

    print(f"\n{'Day':>4} {'Action':<25} {'Reward':>8} {'Budget':>10} {'Nutrition Sum':>14}")
    print("─" * 67)

    for step in range(MAX_DAYS):
        action            = predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward     += reward
        alert             = bool((info["prices"] > BASE_PRICES * 1.3).any())

        frame = renderer.render_frame(info, action, reward, total_reward, alert)
        frames.append(frame)

        nutr_sum = info["nutrition"].sum()
        print(f"  {info['day']:>2} | {_action_name(action):<23} | {reward:>+6.2f} | "
              f"{info['budget']:>9.0f} | {nutr_sum:>12.2f}")

        if terminated or truncated:
            break

    env.close()
    renderer.close()

    import imageio
    imageio.mimsave(save_path, frames, fps=6, loop=0)
    print(f"\n{'═'*55}")
    print(f"  Total reward  : {total_reward:.2f}")
    print(f"  Budget left   : {info['budget']:.0f} RWF")
    print(f"  Nutrition sum : {info['nutrition'].sum():.2f}")
    print(f"  Demo saved    : {save_path}")
    print(f"{'═'*55}")

    # JSON serialisation 
    summary = {
        "algorithm":     algo,
        "total_reward":  round(total_reward, 2),
        "budget_left":   round(float(info["budget"]), 2),
        "nutrition":     {
            "protein":  round(float(info["nutrition"][0]), 2),
            "carbs":    round(float(info["nutrition"][1]), 2),
            "vitamins": round(float(info["nutrition"][2]), 2),
        },
        "purchases":     info["purchases"],
    }
    with open("agent_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON summary  : agent_summary.json")
    print(f"\n  JSON output ready for API / web integration")
    return summary

#  CLI
def main():
    parser = argparse.ArgumentParser(description="SokoPrice RL Demo")
    parser.add_argument("--algo",   choices=["dqn", "ppo", "reinforce", "auto"], default="auto")
    parser.add_argument("--model",  type=str, default=None, help="Path to model file")
    parser.add_argument("--random", action="store_true",  help="Run random agent demo only")
    parser.add_argument("--show",   action="store_true",  help="Display window (local use only)")
    parser.add_argument("--3d",     action="store_true",  dest="use_3d", help="Use 3D renderer")
    args = parser.parse_args()

    headless = not args.show

    if args.random:
        if args.use_3d:
            from environment.rendering_3d import run_random_agent_demo
            run_random_agent_demo("random_agent_demo.gif")
        else:
            run_random_agent_demo(save_path="random_agent_demo.gif", headless=headless)
        return

    algo       = args.algo
    model_path = args.model

    if model_path is None:
        print("\n  Auto-detecting best model from experiment logs …")
        algo, model_path = find_best_model()
        if algo is None:
            print("  No trained models found. Run training scripts first, or use --random.")
            sys.exit(1)

    run_demo(algo, model_path, headless=headless)


if __name__ == "__main__":
    main()
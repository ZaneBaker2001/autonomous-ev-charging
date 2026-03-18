from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch

from agents.ppo_agent import PPOAgent, PPOConfig
from envs.ev_charging_env import EVChargingEnv, EVChargingConfig
from utils.seed import set_seed


def run_policy(env: EVChargingEnv, agent: PPOAgent, episodes: int = 5):
    episode_returns = []
    last_traj = None

    for ep in range(episodes):
        obs, _ = env.reset(seed=100 + ep)
        done = False
        ep_return = 0.0

        traj = {
            "price": [],
            "solar_kw": [],
            "demand_kw": [],
            "served_kw": [],
            "unmet_kw": [],
            "grid_import_kw": [],
            "grid_export_kw": [],
            "battery_soc": [],
            "reward": [],
        }

        while not done:
            action, _, _ = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward

            traj["price"].append(info["price"])
            traj["solar_kw"].append(info["solar_kw"])
            traj["demand_kw"].append(info["demand_kw"])
            traj["served_kw"].append(info["actual_ev_served_kw"])
            traj["unmet_kw"].append(info["unmet_kw"])
            traj["grid_import_kw"].append(info["grid_import_kw"])
            traj["grid_export_kw"].append(info["grid_export_kw"])
            traj["battery_soc"].append(info["battery_soc"])
            traj["reward"].append(info["reward"])

        episode_returns.append(ep_return)
        last_traj = traj
        print(f"Episode {ep + 1}: return={ep_return:.3f}")

    print(f"\nMean return over {episodes} eval episodes: {np.mean(episode_returns):.3f}")
    return last_traj


def plot_trajectory(traj: dict):
    x = np.arange(len(traj["price"])) / 4.0

    plt.figure(figsize=(12, 5))
    plt.plot(x, traj["demand_kw"], label="EV demand (kW)")
    plt.plot(x, traj["served_kw"], label="Served (kW)")
    plt.plot(x, traj["solar_kw"], label="Solar (kW)")
    plt.xlabel("Hour of day")
    plt.ylabel("kW")
    plt.title("Demand, Served Load, and Solar")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(x, traj["price"], label="Grid price ($/kWh)")
    plt.xlabel("Hour of day")
    plt.ylabel("Price")
    plt.title("Dynamic Electricity Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(x, traj["battery_soc"], label="Battery SOC")
    plt.xlabel("Hour of day")
    plt.ylabel("SOC")
    plt.title("Battery State of Charge")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(x, traj["grid_import_kw"], label="Grid import (kW)")
    plt.plot(x, traj["grid_export_kw"], label="Grid export (kW)")
    plt.xlabel("Hour of day")
    plt.ylabel("kW")
    plt.title("Grid Exchange")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(x, traj["unmet_kw"], label="Unmet charging demand (kW)")
    plt.xlabel("Hour of day")
    plt.ylabel("kW")
    plt.title("Service Quality")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    set_seed(42)

    env = EVChargingEnv(EVChargingConfig(), seed=42)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = PPOAgent(
        PPOConfig(
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=device,
        )
    )
    agent.load("checkpoints/ppo_ev_charging_final.pt")

    traj = run_policy(env, agent, episodes=5)
    plot_trajectory(traj)


if __name__ == "__main__":
    main()
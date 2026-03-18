from __future__ import annotations

import os
from collections import deque

import numpy as np
import torch

from agents.ppo_agent import PPOAgent, PPOConfig, RolloutBuffer
from envs.ev_charging_env import EVChargingEnv, EVChargingConfig
from utils.seed import set_seed


def main():
    seed = 42
    set_seed(seed)

    env = EVChargingEnv(EVChargingConfig(), seed=seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = PPOAgent(
        PPOConfig(
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=device,
            steps_per_rollout=4096,
            minibatch_size=256,
            train_iters=10,
        )
    )

    total_timesteps = 150_000
    rollout_steps = agent.cfg.steps_per_rollout
    buffer = RolloutBuffer()

    obs, _ = env.reset()
    episode_return = 0.0
    episode_len = 0
    episode_returns = deque(maxlen=30)

    os.makedirs("checkpoints", exist_ok=True)

    timesteps = 0
    update_idx = 0

    while timesteps < total_timesteps:
        buffer.clear()

        for _ in range(rollout_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.add(obs, action, reward, done, log_prob, value)

            obs = next_obs
            episode_return += reward
            episode_len += 1
            timesteps += 1

            if done:
                episode_returns.append(episode_return)
                obs, _ = env.reset()
                episode_return = 0.0
                episode_len = 0

            if timesteps >= total_timesteps:
                break

        with torch.no_grad():
            if terminated or truncated:
                next_value = 0.0
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                _, _, value_t = agent.model.forward(obs_t)
                next_value = float(value_t.item())

        metrics = agent.update(buffer, next_value)
        update_idx += 1

        avg_return = np.mean(episode_returns) if episode_returns else float("nan")

        print(
            f"update={update_idx:03d} "
            f"timesteps={timesteps:07d} "
            f"avg_ep_return={avg_return:8.3f} "
            f"policy_loss={metrics['policy_loss']:.4f} "
            f"value_loss={metrics['value_loss']:.4f} "
            f"entropy={metrics['entropy']:.4f} "
            f"kl={metrics['approx_kl']:.4f}"
        )

        if update_idx % 10 == 0:
            agent.save("checkpoints/ppo_ev_charging.pt")

    agent.save("checkpoints/ppo_ev_charging_final.pt")
    print("Training complete. Model saved to checkpoints/ppo_ev_charging_final.pt")


if __name__ == "__main__":
    main()
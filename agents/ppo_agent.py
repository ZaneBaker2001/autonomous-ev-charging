from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class PPOConfig:
    obs_dim: int
    act_dim: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    train_iters: int = 10
    minibatch_size: int = 256
    steps_per_rollout: int = 4096
    max_grad_norm: float = 0.5
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    device: str = "cpu"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        self.mu_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared(obs)
        mu = self.mu_head(h)
        value = self.value_head(h).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std, value

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, std, value = self.forward(obs)
        dist = Normal(mu, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        # Tanh-squashed log-prob correction
        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)

        return action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, std, value = self.forward(obs)

        # Approx inverse tanh for already squashed actions
        clipped_actions = torch.clamp(actions, -0.999, 0.999)
        raw_actions = 0.5 * torch.log((1 + clipped_actions) / (1 - clipped_actions))

        dist = Normal(mu, std)
        log_prob = dist.log_prob(raw_actions).sum(-1)
        log_prob -= torch.log(1 - clipped_actions.pow(2) + 1e-6).sum(-1)

        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


class RolloutBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        self.obs.append(obs.copy())
        self.actions.append(action.copy())
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))

    def clear(self) -> None:
        self.__init__()


class PPOAgent:
    def __init__(self, config: PPOConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        self.model = ActorCritic(config.obs_dim, config.act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t, log_prob_t, value_t = self.model.act(obs_t)
        action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
        return action, float(log_prob_t.item()), float(value_t.item())

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_nonterminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_nonterminal = 1.0 - dones[t]
                next_values = values[t + 1]

            delta = rewards[t] + self.cfg.gamma * next_values * next_nonterminal - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, buffer: RolloutBuffer, next_value: float) -> dict:
        obs = np.array(buffer.obs, dtype=np.float32)
        actions = np.array(buffer.actions, dtype=np.float32)
        rewards = np.array(buffer.rewards, dtype=np.float32)
        dones = np.array(buffer.dones, dtype=np.float32)
        old_log_probs = np.array(buffer.log_probs, dtype=np.float32)
        values = np.array(buffer.values, dtype=np.float32)

        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        n = len(obs)
        idxs = np.arange(n)

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }

        num_updates = 0

        for _ in range(self.cfg.train_iters):
            np.random.shuffle(idxs)

            for start in range(0, n, self.cfg.minibatch_size):
                mb_idx = idxs[start:start + self.cfg.minibatch_size]

                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_adv = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                new_log_probs, entropy, values_pred = self.model.evaluate_actions(mb_obs, mb_actions)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio
                )

                policy_loss = -torch.min(ratio * mb_adv, clipped_ratio * mb_adv).mean()
                value_loss = ((values_pred - mb_returns) ** 2).mean()
                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                approx_kl = (mb_old_log_probs - new_log_probs).mean().item()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy_bonus.item())
                metrics["approx_kl"] += float(approx_kl)
                num_updates += 1

        for key in metrics:
            metrics[key] /= max(num_updates, 1)

        return metrics

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
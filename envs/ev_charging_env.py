from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class EVChargingConfig:
    episode_steps: int = 96  # 24h at 15-min resolution
    battery_capacity_kwh: float = 300.0
    battery_max_charge_kw: float = 120.0
    battery_max_discharge_kw: float = 120.0
    battery_efficiency: float = 0.96
    grid_import_limit_kw: float = 250.0
    grid_export_limit_kw: float = 120.0
    charger_limit_kw: float = 180.0
    initial_battery_soc: float = 0.5

    # Reward economics
    unmet_demand_penalty: float = 1.8
    battery_degradation_cost: float = 0.03
    export_discount: float = 0.8
    action_smoothing_penalty: float = 0.01


class EVChargingEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: EVChargingConfig | None = None, seed: int | None = None):
        super().__init__()
        self.cfg = config or EVChargingConfig()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        self.rng = np.random.default_rng(seed)
        self.seed_value = seed

        self.t = 0
        self.battery_soc = self.cfg.initial_battery_soc
        self.prev_action = np.zeros(3, dtype=np.float32)

        self.price_profile = np.zeros(self.cfg.episode_steps, dtype=np.float32)
        self.solar_profile = np.zeros(self.cfg.episode_steps, dtype=np.float32)
        self.demand_profile = np.zeros(self.cfg.episode_steps, dtype=np.float32)

    def _generate_daily_profiles(self) -> None:
        steps = self.cfg.episode_steps
        hours = np.arange(steps) / 4.0

        # Time-of-use price with morning/evening peaks + noise
        base_price = 0.12 + 0.05 * np.sin((hours - 6) * np.pi / 12) ** 2
        evening_peak = 0.18 * np.exp(-0.5 * ((hours - 18.5) / 2.2) ** 2)
        morning_peak = 0.08 * np.exp(-0.5 * ((hours - 8.0) / 1.8) ** 2)
        price_noise = self.rng.normal(0.0, 0.01, size=steps)
        self.price_profile = np.clip(
            base_price + evening_peak + morning_peak + price_noise, 0.05, 0.45
        ).astype(np.float32)

        # Solar generation peaks around midday
        solar_shape = np.exp(-0.5 * ((hours - 13.0) / 3.0) ** 2)
        weather_factor = self.rng.uniform(0.55, 1.05)
        cloud_noise = self.rng.normal(0.0, 6.0, size=steps)
        self.solar_profile = np.clip(
            140.0 * solar_shape * weather_factor + cloud_noise, 0.0, 160.0
        ).astype(np.float32)

        # EV demand: commute peaks + randomness + occasional site event spikes
        morning_demand = 60.0 * np.exp(-0.5 * ((hours - 8.5) / 1.8) ** 2)
        evening_demand = 85.0 * np.exp(-0.5 * ((hours - 18.0) / 2.3) ** 2)
        midday_demand = 30.0 * np.exp(-0.5 * ((hours - 13.0) / 2.5) ** 2)

        base = 18.0 + 6.0 * np.sin((hours - 5.0) * np.pi / 12) ** 2
        noise = self.rng.normal(0.0, 8.0, size=steps)

        demand = base + morning_demand + evening_demand + midday_demand + noise

        # Add one random local event spike
        event_center = self.rng.integers(28, 84)
        event_magnitude = self.rng.uniform(15.0, 45.0)
        event_width = self.rng.uniform(2.0, 5.0)
        demand += event_magnitude * np.exp(
            -0.5 * ((np.arange(steps) - event_center) / event_width) ** 2
        )

        self.demand_profile = np.clip(demand, 5.0, self.cfg.charger_limit_kw).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        idx = min(self.t, self.cfg.episode_steps - 1)
        theta = 2.0 * np.pi * idx / self.cfg.episode_steps

        window_start = max(0, idx - 7)
        rolling_avg_price = float(np.mean(self.price_profile[window_start:idx + 1]))
        rolling_avg_demand = float(np.mean(self.demand_profile[window_start:idx + 1]))

        obs = np.array(
            [
                np.sin(theta),
                np.cos(theta),
                self.price_profile[idx],
                self.solar_profile[idx],
                self.demand_profile[idx],
                self.battery_soc,
                self.prev_action[0],
                self.prev_action[1],
                rolling_avg_price,
                rolling_avg_demand / max(self.cfg.charger_limit_kw, 1.0),
            ],
            dtype=np.float32,
        )
        return obs

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        self.battery_soc = self.cfg.initial_battery_soc + self.rng.uniform(-0.05, 0.05)
        self.battery_soc = float(np.clip(self.battery_soc, 0.1, 0.9))
        self.prev_action = np.zeros(3, dtype=np.float32)

        self._generate_daily_profiles()
        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        idx = self.t
        dt_hours = 0.25

        price = float(self.price_profile[idx])
        solar_kw = float(self.solar_profile[idx])
        demand_kw = float(self.demand_profile[idx])

        battery_signal = float(action[0])
        charge_fraction = float((action[1] + 1.0) / 2.0)
        grid_bias = float(action[2])

        target_ev_served_kw = min(demand_kw * charge_fraction, self.cfg.charger_limit_kw)

        # Battery dispatch
        if battery_signal >= 0:
            battery_charge_kw = battery_signal * self.cfg.battery_max_charge_kw
            battery_discharge_kw = 0.0
        else:
            battery_discharge_kw = (-battery_signal) * self.cfg.battery_max_discharge_kw
            battery_charge_kw = 0.0

        battery_energy_kwh = self.battery_soc * self.cfg.battery_capacity_kwh

        # Enforce charge feasibility
        max_charge_possible_kwh = self.cfg.battery_capacity_kwh - battery_energy_kwh
        feasible_charge_kw = min(
            battery_charge_kw,
            max_charge_possible_kwh / (dt_hours * self.cfg.battery_efficiency + 1e-8),
        )

        # Enforce discharge feasibility
        max_discharge_possible_kwh = battery_energy_kwh
        feasible_discharge_kw = min(
            battery_discharge_kw,
            (max_discharge_possible_kwh * self.cfg.battery_efficiency) / (dt_hours + 1e-8),
        )

        battery_charge_kw = max(0.0, feasible_charge_kw)
        battery_discharge_kw = max(0.0, feasible_discharge_kw)

        # Site power balance
        # supply = solar + battery_discharge + grid_import
        # uses   = ev_served + battery_charge + grid_export
        net_without_grid = solar_kw + battery_discharge_kw - target_ev_served_kw - battery_charge_kw

        # Interpret grid bias:
        # if surplus exists, negative bias encourages export; if deficit exists, positive bias encourages import.
        if net_without_grid >= 0:
            desired_export_kw = net_without_grid * ((1.0 - grid_bias) / 2.0)
            grid_export_kw = min(desired_export_kw, self.cfg.grid_export_limit_kw)
            grid_import_kw = 0.0
        else:
            desired_import_kw = (-net_without_grid) * ((1.0 + grid_bias) / 2.0)
            grid_import_kw = min(desired_import_kw, self.cfg.grid_import_limit_kw)
            grid_export_kw = 0.0

        realized_net = solar_kw + battery_discharge_kw + grid_import_kw - target_ev_served_kw - battery_charge_kw - grid_export_kw

        # If still surplus, curtail.
        curtailed_solar_kw = max(0.0, realized_net)

        # If still deficit, EV demand is unmet.
        unmet_kw = max(0.0, -realized_net)
        actual_ev_served_kw = max(0.0, target_ev_served_kw - unmet_kw)

        # Update battery SOC
        battery_energy_kwh += battery_charge_kw * dt_hours * self.cfg.battery_efficiency
        battery_energy_kwh -= (battery_discharge_kw * dt_hours) / self.cfg.battery_efficiency
        battery_energy_kwh = float(np.clip(battery_energy_kwh, 0.0, self.cfg.battery_capacity_kwh))
        self.battery_soc = battery_energy_kwh / self.cfg.battery_capacity_kwh

        # Economics
        import_cost = grid_import_kw * dt_hours * price
        export_revenue = grid_export_kw * dt_hours * price * self.cfg.export_discount
        unmet_penalty = unmet_kw * dt_hours * self.cfg.unmet_demand_penalty
        battery_deg_cost = (
            (battery_charge_kw + battery_discharge_kw) * dt_hours * self.cfg.battery_degradation_cost
        )
        smoothing_penalty = (
            np.mean(np.abs(action - self.prev_action)) * self.cfg.action_smoothing_penalty
        )

        reward = -(
            import_cost
            + unmet_penalty
            + battery_deg_cost
            + smoothing_penalty
            - export_revenue
        )

        info = {
            "price": price,
            "solar_kw": solar_kw,
            "demand_kw": demand_kw,
            "target_ev_served_kw": target_ev_served_kw,
            "actual_ev_served_kw": actual_ev_served_kw,
            "battery_charge_kw": battery_charge_kw,
            "battery_discharge_kw": battery_discharge_kw,
            "grid_import_kw": grid_import_kw,
            "grid_export_kw": grid_export_kw,
            "curtailed_solar_kw": curtailed_solar_kw,
            "unmet_kw": unmet_kw,
            "battery_soc": self.battery_soc,
            "import_cost": import_cost,
            "export_revenue": export_revenue,
            "reward": reward,
        }

        self.prev_action = action.copy()
        self.t += 1

        terminated = self.t >= self.cfg.episode_steps
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        idx = min(self.t, self.cfg.episode_steps - 1)
        print(
            f"t={idx:02d} "
            f"price={self.price_profile[idx]:.3f} "
            f"solar={self.solar_profile[idx]:.1f}kW "
            f"demand={self.demand_profile[idx]:.1f}kW "
            f"soc={self.battery_soc:.2f}"
        )
# Autonomous energy management system for EV charging sites


## Overview

Modern EV charging sites are increasingly deployed with local solar and battery storage. Operating these systems well is hard because the controller must continuously balance competing objectives:

- Serve as much EV demand as possible
- Reduce costly grid imports during high-price periods
- Use the battery strategically
- Export excess solar when profitable
- Avoid excessive battery cycling
- Remain smooth and stable in control decisions

This project formulates that problem as a **continuous-control reinforcement learning task** and trains a **Proximal Policy Optimization (PPO)** agent to manage the site over a daily operating horizon.

Each episode represents a full day at **15-minute resolution**. At each timestep, the agent chooses:

1. Battery dispatch
2. Charging allocation
3. Grid exchange preference

The environment then simulates site power balance, cost/revenue, battery state of charge, and service quality.

---

## Problem formulation

### Objective

Train an RL agent to operate an EV charging site that minimizes total operating cost while maintaining service quality.

### System components

The simulated site includes:

- **EV charging demand**
- **solar generation**
- **grid import/export**
- **battery storage**

### Trade-offs the agent learns

The agent must learn when to:

- Charge the battery using cheap energy or solar surplus
- Discharge the battery to reduce expensive grid purchases
- Serve more or less instantaneous EV demand depending on economics and constraints
- Export surplus solar energy
- Avoid unstable or overly aggressive control actions

---

## Environment design

Each episode is one day:

- **96 timesteps**
- **15 minutes per step**
- **24 hours total**

The environment creates stochastic daily profiles for:

- electricity price,
- solar generation,
- EV demand.

This means the policy must learn a **general operating strategy**, not memorize a single deterministic schedule.

### Observation space

The state includes:

- sine/cosine time-of-day encoding,
- current electricity price,
- current solar generation,
- current EV charging demand,
- battery state of charge,
- previous battery action,
- previous charging action,
- rolling average price,
- rolling average demand.

These features give the agent both instantaneous context and some short-term temporal signal.

### Action space

The agent outputs 3 continuous actions in `[-1, 1]`:

1. **Battery dispatch**
   - `-1`: maximum discharge
   - `+1`: maximum charge

2. **Charging allocation factor**
   - `-1`: serve none of current demand
   - `+1`: serve all current demand

3. **Grid exchange bias**
   - `-1`: prefer export when surplus exists
   - `+1`: prefer import when deficit exists

### Reward function

The reward is designed around real operational economics.

The agent is penalized for:

- Grid import cost
- Unmet EV charging demand
- Battery degradation from cycling
- Abrupt action changes

The agent is rewarded through:

- export revenue when excess energy is sold back to the grid.

Formally, reward is the negative of:

- Import cost
- Unmet demand penalty
- Battery degradation cost
- Smoothing penalty

plus export revenue.

This gives a meaningful optimization objective: operate the site profitably while maintaining charging service.

---

## PPO agent

The learning algorithm is **Proximal Policy Optimization (PPO)** implemented directly in PyTorch.

### Model structure

The actor-critic network uses:

- Shared MLP encoder
- Policy head for continuous actions
- Value head for state value estimation
- Learned log standard deviation for Gaussian exploration

### PPO details

The implementation includes:

- Clipped policy objective
- Generalized advantage estimation (GAE)
- Entropy bonus
- Value loss
- Gradient clipping
- Tanh-squashed continuous actions

---

## Repository structure

```text
autonomous-ev-charging/
├── README.md
├── requirements.txt
├── train.py
├── evaluate.py
├── envs/
│   └── ev_charging_env.py
├── agents/
│   └── ppo_agent.py
└── utils/
    └── seed.py
```

## Installation 

To create a virtual environment on MacOS or Linux: 

```bash
python3 -m venv .venv
source .venv/bin/activate
```

To create a virtual environment on Windows:
```bash
python3 -m venv .venv
.venv\Scripts\activate
```

To install the required packages: 
```bash
pip3 install -r requirements.txt 
```

## Training Instructions 

To begin training the model:

```bash
python3 train.py
```

Doing this will:

- Initialize the custom EV charging environment
- Train a PPO agent
- Print update metrics during training
- Save checkpoints in checkpoints/

During training, checkpoints are saved to:

```text
checkpoints/ppo_ev_charging.pt
```

The final model is saved to: 

```text
checkpoints/ppo_ev_charging_final.pt
```

## Evaluation Instructions 

To evaluate the final model:

```bash
python3 evaluate.py
``` 

Doing this will:
	
- Load the trained model
- Evaluate it over multiple episodes
- Print episodic returns
- Generate plots 

## Visualizations 

The generated plots visualize:

- Demand vs served load vs solar
- Electricity price
- Battery state of charge
- Grid import/export
- Unmet charging demand

## Behavior Analysis 

The below graphs illustrate the agent's behavior across a variety of categories:

![Service Quality](./service_quality_sample.png)
![Grid Exchange](./grid_exchange_sample.png) 
![Battery State of Charge](./battery_state_of_charge_sample.png) 
![Dynamic Electricity Price](./dynamic_electricity_price_sample.png)
![Demand, Served Load, and Solar](./demand_served_load_solar_sample.png)

The key takeaways from these graphs are:

- The agent learns to focus on the economics of the system, often selling extra solar power back to the grid and steering clear of costly grid purchases.
- At the same time, its planning over time is still limited. It tends to use up the battery too early instead of saving or recharging it for later periods. 
- That points to a gap in the current reward design: it does not seem to encourage energy arbitrage strongly enough, which makes this an important area to improve.
- Even so, overall service remains strong, with unmet demand appearing only in short-lived spikes.

## Results 

Training was run for 150,000 timesteps.


Key training milestones included:

| Update | Timesteps | Avg Episode Return |
|---|---:|---:|
| 1 | 4,096 | -798.859 |
| 5 | 20,480 | -436.741 |
| 10 | 40,960 | -225.108 |
| 15 | 61,440 | -148.365 |
| 20 | 81,920 | -40.860 |
| 25 | 102,400 | 13.153 |
| 30 | 122,880 | 49.689 |
| 35 | 143,360 | 71.989 |
| 37 | 150,000 | 83.403 |

The PPO agent improved substantially over training:

- Early training began with very poor returns, around -799 average episodic return.
- Returns steadily improved as the agent learned basic operating behavior.
- Performance crossed into positive average return by update 25.
- Final training reached 83.403 average episodic return by the last update.






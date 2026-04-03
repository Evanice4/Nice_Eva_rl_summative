"""
SokoPrice Custom Gymnasium Environment
Mission: Empower Rwandan households with price intelligence and nutrition planning
         in informal agricultural markets.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


#  Market constants
FOOD_ITEMS = [
    "Beans",      # 0
    "Maize",      # 1
    "Cassava",    # 2
    "Banana",     # 3
    "Tomato",     # 4
    "Spinach",    # 5
    "Sweet Potato",# 6
    "Rice",       # 7
]
N_ITEMS = len(FOOD_ITEMS)

# Nutritional value per unit (protein, carbs, vitamins) - normalised 0‑1
NUTRITION = np.array([
    [0.9, 0.6, 0.3],   # Beans
    [0.3, 0.9, 0.2],   # Maize
    [0.2, 0.8, 0.3],   # Cassava
    [0.1, 0.7, 0.5],   # Banana
    [0.1, 0.2, 0.9],   # Tomato
    [0.4, 0.1, 0.9],   # Spinach
    [0.2, 0.8, 0.5],   # Sweet Potato
    [0.3, 0.9, 0.1],   # Rice
], dtype=np.float32)

# Base prices (RWF per unit) and daily volatility (std dev)
BASE_PRICES  = np.array([500, 300, 250, 200, 400, 350, 280, 600], dtype=np.float32)
PRICE_SIGMA  = np.array([ 80,  50,  40,  30,  90,  60,  45, 100], dtype=np.float32)

# Minimum daily nutritional targets [protein, carbs, vitamins]
NUTRITION_TARGET = np.array([1.5, 2.0, 1.5], dtype=np.float32)

DAILY_BUDGET   = 3000.0   # RWF per day
MAX_DAYS       = 14        # Episode length (2-week planning horizon)
MAX_UNITS      = 5         # Max units buyable per item per action


#  Action encoding
# Actions:
#   0..N_ITEMS-1           -> Buy 1 unit of item i
#   N_ITEMS..2*N_ITEMS-1   -> Buy 2 units of item i
#   2*N_ITEMS..3*N_ITEMS-1 -> Buy 3 units of item i
#   3*N_ITEMS              -> Skip (do not buy today)
#   3*N_ITEMS + 1          -> Issue price alert (flag high-price day)
#   3*N_ITEMS + 2          -> Switch to cheaper substitute (heuristic)
N_BUY_TIERS = 3
N_ACTIONS   = N_BUY_TIERS * N_ITEMS + 3   # = 27


class SokoPriceEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Spaces 
        self.observation_space = spaces.Box(
            low   = np.zeros(25, dtype=np.float32),
            high  = np.ones(25,  dtype=np.float32),
            dtype = np.float32,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # internal state
        self._prices_prev  = None
        self._prices       = None
        self._nutrition    = None
        self._budget       = None
        self._day          = None
        self._alert_active = None
        self._purchase_log = None
        self._item_counts  = None   # track how many times each item bought

    #  Helpers 
    def _sample_prices(self):
        """Sample today's market prices with random walk."""
        noise = np.random.randn(N_ITEMS).astype(np.float32) * PRICE_SIGMA
        prices = np.clip(self._prices + noise, BASE_PRICES * 0.5, BASE_PRICES * 2.0)
        return prices

    def _get_obs(self):
        price_norm  = self._prices / (BASE_PRICES * 2.0)
        trend_norm  = np.clip((self._prices - self._prices_prev) / (PRICE_SIGMA + 1e-6), -1, 1) * 0.5 + 0.5
        nutr_norm   = np.clip(self._nutrition / (NUTRITION_TARGET * MAX_DAYS), 0, 1)
        budget_norm = np.clip(self._budget / DAILY_BUDGET, 0, 1)
        day_norm    = self._day / MAX_DAYS
        deficit     = np.clip((NUTRITION_TARGET - self._nutrition / max(self._day, 1)) / NUTRITION_TARGET, 0, 1)
        alert       = np.array([float(self._alert_active)], dtype=np.float32)

        obs = np.concatenate([
            price_norm, trend_norm, nutr_norm,
            [budget_norm], [day_norm],
            deficit, alert
        ]).astype(np.float32)
        return obs

    def _info(self):
        return {
            "day":       self._day,
            "budget":    self._budget,
            "nutrition": self._nutrition.copy(),
            "prices":    self._prices.copy(),
            "purchases": self._purchase_log.copy(),
        }

    # Core API 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._prices_prev  = BASE_PRICES.copy()
        self._prices       = BASE_PRICES + np.random.randn(N_ITEMS).astype(np.float32) * PRICE_SIGMA * 0.3
        self._prices       = np.clip(self._prices, BASE_PRICES * 0.5, BASE_PRICES * 2.0)
        self._nutrition    = np.zeros(3, dtype=np.float32)
        self._budget       = DAILY_BUDGET * MAX_DAYS   # total episode budget
        self._day          = 0
        self._alert_active = False
        self._purchase_log = []
        self._item_counts  = np.zeros(N_ITEMS, dtype=np.int32)
        return self._get_obs(), self._info()

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"
        reward = 0.0
        terminated = False
        truncated  = False

        self._day += 1
        self._prices_prev = self._prices.copy()
        self._prices      = self._sample_prices()

        #  Decode action and apply its effects 
        if action < N_BUY_TIERS * N_ITEMS:
            tier      = action // N_ITEMS        # 0,1,2 → 1,2,3 units
            item_idx  = action % N_ITEMS
            units     = tier + 1
            cost      = self._prices[item_idx] * units

            if cost <= self._budget:
                self._budget    -= cost
                gained_nutr      = NUTRITION[item_idx] * units
                self._nutrition += gained_nutr
                nutr_gain        = gained_nutr.sum()
                budget_saved     = (DAILY_BUDGET - cost / units) / DAILY_BUDGET
                # Diversity penalty: discourage buying same item every day
                repeat_penalty   = 0.5 * max(0, self._item_counts[item_idx] - 2)
                self._item_counts[item_idx] += 1
                reward          += nutr_gain * 2.0 + budget_saved * 0.5 - repeat_penalty
                self._purchase_log.append({
                    "day": self._day, "item": FOOD_ITEMS[item_idx],
                    "units": units, "cost": float(cost)
                })
            else:
                reward -= 2.0  # tried to buy but over budget

        elif action == N_BUY_TIERS * N_ITEMS:   # skip
            reward += 0.1   # small bonus for conserving budget wisely

        elif action == N_BUY_TIERS * N_ITEMS + 1:  # alert
            # Detect if any price is > 1.3× base → spike
            spike = np.any(self._prices > BASE_PRICES * 1.3)
            if spike:
                reward += 1.5   # correctly flagged a spike
                self._alert_active = True
            else:
                reward -= 0.5   # false alarm
                self._alert_active = False

        elif action == N_BUY_TIERS * N_ITEMS + 2:  # substitute
            # Buy 1 unit of the cheapest item relative to nutrition value
            value_ratio = (NUTRITION.sum(axis=1) + 1e-6) / (self._prices + 1e-6)
            best        = int(np.argmax(value_ratio))
            cost        = self._prices[best]
            if cost <= self._budget:
                self._budget    -= cost
                gained_nutr      = NUTRITION[best]
                self._nutrition += gained_nutr
                reward          += gained_nutr.sum() * 2.0 + 0.3  # bonus for smart substitution
                self._purchase_log.append({
                    "day": self._day, "item": FOOD_ITEMS[best],
                    "units": 1, "cost": float(cost)
                })
            else:
                reward -= 1.0

        #  Terminal conditions
        if self._day >= MAX_DAYS:
            terminated = True
            # Check if nutrition targets met
            avg_daily_nutr = self._nutrition / MAX_DAYS
            deficit        = np.maximum(0, NUTRITION_TARGET - avg_daily_nutr)
            reward        -= deficit.sum() * 3.0   # penalise nutritional deficit
            if deficit.sum() < 0.3 and self._budget >= 0:
                reward += 5.0   # bonus: healthy + within budget

        if self._budget < 0:
            terminated = True
            reward    -= 5.0   # hard penalty for bankruptcy

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, self._info()

    def render(self):
        """Returns an RGB array - actual rendering done in rendering.py"""
        pass

    def close(self):
        pass
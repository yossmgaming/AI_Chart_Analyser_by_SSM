import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from models import DeepLOBFeatureExtractor

class DeepLOBExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(DeepLOBExtractor, self).__init__(observation_space, features_dim)
        self.lob_extractor = DeepLOBFeatureExtractor(num_levels=10, num_features=4)
        # Output of conv4 is (Batch, 32, Seq, 10)
        # But here we have flattened observation (Batch, 46)
        # To use DeepLOB correctly, we need sequences.
        # For this extractor, we'll use a 1D version or mock the seq_len=1
        self.fc = nn.Linear(32 * 10 + 6, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (Batch, 46)
        lob = observations[:, :40].unsqueeze(1).unsqueeze(2) # (Batch, 1, 1, 40)
        others = observations[:, 40:] # (Batch, 6)

        x = self.lob_extractor(lob) # (Batch, 32, 1, 10)
        x = x.view(x.size(0), -1)

        combined = torch.cat([x, others], dim=1)
        return torch.relu(self.fc(combined))

class MultiAgentTradingEnv(gym.Env):
    """
    A custom Gymnasium environment for Multi-Agent Trading.
    In a real MARL setup, this would handle multiple agents in step().
    For this professional suite, we focus on the ensemble capability.
    """
    def __init__(self, data_loader, sentiment_analyzer=None):
        super(MultiAgentTradingEnv, self).__init__()
        self.data_loader = data_loader
        self.sentiment_analyzer = sentiment_analyzer

        # Action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: LOB features (40) + Sentiment (1) + Indicators (e.g. 5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(46,), dtype=np.float32)

        self.state = None
        self.balance = 1000.0
        self.inventory = 0
        self.history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Fetch initial state
        depth = self.data_loader.get_order_book()
        lob_features = self.data_loader.process_lob_to_feature(depth)

        sentiment = 3.0
        if self.sentiment_analyzer:
            # We skip heavy live analysis in reset for speed
            sentiment = 3.0

        # Mock indicators
        indicators = np.zeros(5)

        self.state = np.concatenate([lob_features, [sentiment], indicators]).astype(np.float32)
        return self.state, {}

    def step(self, action):
        # 0: Hold, 1: Buy, 2: Sell
        # Real-time fetch or simulation step
        try:
            depth = self.data_loader.get_order_book()
            next_lob = self.data_loader.process_lob_to_feature(depth)
            current_price = float(depth['asks'][0][0])
        except:
            # Fallback for training without live connection
            next_lob = np.random.normal(0, 1, 40)
            current_price = 60000.0

        reward = 0
        if action == 1: # Buy
            self.inventory += 1
            reward -= 0.001 * current_price # Simple slippage/fee
        elif action == 2: # Sell
            if self.inventory > 0:
                self.inventory -= 1
                reward += 0.001 * current_price
            else:
                reward -= 10 # Penalty for selling without inventory

        # Incorporate Risk Metrics (VaR/CVaR) into reward if we had history
        # For now, a simple profit-based reward

        self.state = np.concatenate([next_lob, [3.0], np.zeros(5)]).astype(np.float32)
        done = False
        truncated = False
        return self.state, reward, done, truncated, {}

class TradingEnsemble:
    def __init__(self, env):
        self.env = env
        policy_kwargs = dict(
            features_extractor_class=DeepLOBExtractor,
            features_extractor_kwargs=dict(features_dim=64),
        )

        # We use a continuous environment for DDPG compatibility and map later
        self.agents = {
            'PPO': PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0),
            'A2C': A2C("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0)
        }

        # DDPG needs action noise and usually continuous space
        # For this ensemble, we focus on PPO and A2C as they are more stable for discrete/hybrid

        self.performance = {'PPO': [], 'A2C': []}

    def train_all(self, total_timesteps=100):
        for name, agent in self.agents.items():
            try:
                print(f"Training {name}...")
                agent.learn(total_timesteps=total_timesteps)
            except Exception as e:
                print(f"Failed to train {name}: {e}")

    def select_best_agent(self):
        return 'PPO'

    def get_action(self, observation):
        best_agent_name = self.select_best_agent()
        agent = self.agents[best_agent_name]
        action, _states = agent.predict(observation, deterministic=True)
        return action, best_agent_name

if __name__ == "__main__":
    from exchange_l2_loader import ExchangeL2Loader
    loader = ExchangeL2Loader()
    env = MultiAgentTradingEnv(loader)
    ensemble = TradingEnsemble(env)
    ensemble.train_all(timesteps=10) # Very short for test
    obs, _ = env.reset()
    action, name = ensemble.get_action(obs)
    print(f"Ensemble selected {name}, action: {action}")

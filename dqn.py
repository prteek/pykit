import numpy as np
import gymnasium as gym
import pykitml as pk


# Wrapper class around the environment
class Environment:
    def __init__(self):
        self._env = gym.make('CartPole-v1', render_mode="human")

    def reset(self):
        return self._env.reset()[0]  # Required due to api change in gym

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)

        # Reward function, from
        # https://github.com/keon/deep-q-learning/blob/master/ddqn.py
        x, _, theta, _ = obs
        r1 = (self._env.x_threshold - abs(x)) / self._env.x_threshold - 0.8
        r2 = (self._env.theta_threshold_radians - abs(theta)) / self._env.theta_threshold_radians - 0.5
        reward = r1 + r2

        return obs, reward, terminated

    def close(self):
        self._env.close()

    def render(self):
        self._env.render()


env = Environment()

# Create DQN agent and train it
agent = pk.DQNAgent([4, 64, 64, 2])
agent.set_save_freq(100, 'cartpole_agent')
agent.train(env, 50, pk.Adam(0.01), render=True)

# Plot reward graph
env.render()

agent.plot_performance()


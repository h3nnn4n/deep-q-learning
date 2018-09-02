import gym
import numpy as np
from random_agent import RandomAgent
from collections import deque


class Main:
    def __init__(self):
        self.agent = RandomAgent()
        self.sample_batch_size = 32
        self.max_episodes = 2000

        self.env = gym.make('CartPole-v1')

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.agent.state_size = self.state_size
        self.agent.action_size = self.action_size

        self.scores = deque(maxlen=200)

    def run(self):
        for _ in range(self.max_episodes):
            state = self.env.reset()

            done = False
            total_reward = 0

            while not done:
                action = self.agent.get_action(state)

                state, reward, done, _ = self.env.step(action)

                total_reward += reward

            self.scores.append(total_reward)
            print('%8.2f %8.2f' % (np.mean(self.scores), total_reward))


if __name__ == '__main__':
    main = Main()
    main.run()

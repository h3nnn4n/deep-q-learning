import sys
import gym
import numpy as np
from random_agent import RandomAgent
from deep_q_agent import DeepQAgent
from collections import deque


class Main:
    def __init__(self):
        self.max_episodes = 10000

        self.env = gym.make('CartPole-v1')

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.agent = DeepQAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            fixed_target=True
        )

        self.scores = deque(maxlen=100)
        self.max_score = 0

    def run(self):
        for episode_number in range(self.max_episodes):
            state = self.env.reset()

            done = False
            total_reward = 0

            while not done:
                action = self.agent.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.record(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                self.agent.learn()

            # self.agent.learn()
            self.agent.update_epsilon()

            self.scores.append(total_reward)
            self.max_score = max(self.max_score, total_reward)
            print('%6d %8.2f %8.2f %8.2f %8.2f' % (
                episode_number + 1,
                np.mean(self.scores),
                total_reward,
                self.max_score,
                self.agent.epsilon)
            )

            sys.stdout.flush()


if __name__ == '__main__':
    main = Main()
    main.run()

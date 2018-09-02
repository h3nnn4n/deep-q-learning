import random


class RandomAgent:
    def __init__(self):
        self.state_size = 0
        self.action_size = 0

    def get_action(self, state):
        return random.randrange(self.action_size)

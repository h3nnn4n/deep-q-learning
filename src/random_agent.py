import random


class RandomAgent:
    def __init__(self, state_size=None, action_size=None):
        self.state_size = state_size
        self.action_size = action_size

    def get_action(self, state):
        return random.randrange(self.action_size)

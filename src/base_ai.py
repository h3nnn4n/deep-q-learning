class BaseAI:
    def __init__(self, state_size=None, action_size=None):
        self.state_size = state_size
        self.action_size = action_size

    def on_step(self, state, action, reward, next_state, done):
        pass

    def on_end(self):
        pass

    def on_start(self):
        pass

    def on_init(self):
        pass

    def get_action(self):
        pass

    def save(self, extra=''):
        pass

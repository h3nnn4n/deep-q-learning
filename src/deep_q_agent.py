import random
import numpy as np

import keras
import keras.models
import keras.layers
from keras.models import Sequential
from keras.layers import Dense

from collections import deque


class DeepQAgent:
    def __init__(self, state_size=None, action_size=None, fixed_target=False):
        self.state_size = state_size
        self.action_size = action_size

        self.fixed_target = fixed_target

        self.memory = deque(maxlen=2000)

        self.batch_size = 64

        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.9975

        self.alpha = 0.001
        self.epsilon = self.epsilon_start
        self.gamma = 0.95
        self.tau_iters = 10
        self.tau_count = 0

        self.model = self.build_model()
        self.target_model = self.build_model() if self.fixed_target else None

    def build_model(self):
        model = Sequential()
        model.add(Dense(20, input_dim=self.state_size, activation='relu'))
        model.add(Dense(20, activation='relu'))
        # model.add(Dense(20, activation='relu'))
        # model.add(Dense(10, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.alpha))

        return model

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        return np.argmax(self.model.predict(self.reshape(state)))

    def record(self, state, action, reward, next_state, done):
        state = self.reshape(state)
        next_state = self.reshape(next_state)
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if self.fixed_target:
            self.update_target()

        target_model = self.target_model if self.fixed_target else self.model

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, k=self.batch_size)

        for state, action, reward, next_state, done in batch:
            target = reward

            if not done:
                target = reward + self.gamma * np.amax(
                    target_model.predict(next_state)[0]
                )

            target_f = target_model.predict(state)
            target_f[0][action] = target
            self.model.fit(
                state,
                target_f,
                epochs=1,
                verbose=0
            )

        self.update_epsilon()

    def update_target(self):
        if self.tau_count < self.tau_iters:
            self.tau_count += 1
            return

        self.tau_count = 0
        print('Updated model')

        # I think we can just copy over
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i]

        self.target_model.set_weights(target_weights)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

            self.epsilon = max(self.epsilon, self.epsilon_end)

    def reshape(self, state):
        return np.reshape(state, (1, self.state_size))

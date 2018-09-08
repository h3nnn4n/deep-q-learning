import random
import datetime
import string
import numpy as np

import keras
import keras.models
import keras.layers
from keras.models import Sequential
from keras.layers import Dense

from collections import deque

from base_ai import BaseAI


class DeepQAgent(BaseAI):
    def __init__(self, state_size=None, action_size=None, fixed_target=False):
        self.state_size = state_size
        self.action_size = action_size

        self.fixed_target = fixed_target

        self.memory = deque(maxlen=2000)

        self.batch_size = 16

        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.9975

        self.alpha = 0.0001
        self.epsilon = self.epsilon_start
        self.gamma = 0.99
        self.tau = 0.005

        self.model = self.build_model()
        self.target_model = self.build_model() if self.fixed_target else None

        self.name = 'dqn'
        if self.fixed_target:
            self.name += '_fixed_target'

        self.r_string = None

    def on_start(self):
        self.save_model()

    def on_step(self, state, action, reward, next_state, done):
        self.record(state, action, reward, next_state, done)
        self.learn()

    def on_end(self):
        self.update_epsilon()

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

    def update_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = (
                self.tau * weights[i] +
                (1.0 - self.tau) * target_weights[i]
            )

        self.target_model.set_weights(target_weights)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

            self.epsilon = max(self.epsilon, self.epsilon_end)

    def reshape(self, state):
        return np.reshape(state, (1, self.state_size))

    def save(self, extra=''):
        extra = self.get_suffix() + extra
        base_name_wights = 'model_weights_%s.h5' % extra

        self.target_model.save_weights(base_name_wights)

    def save_model(self):
        extra = self.get_suffix()
        base_name_model = 'model_%s.json' % extra

        with open(base_name_model, 'w') as f:
            f.write(self.target_model.to_json())

    def get_suffix(self):
        now = datetime.datetime.now()
        char_set = string.ascii_uppercase + string.digits

        if self.r_string is None:
            self.r_string = ''.join(random.sample(char_set * 6, 6))

        suffix = "_%s_%04d_%02d_%02d__%02d_%02d_%02d__%s" % (
            self.name,
            now.year,
            now.month,
            now.day,
            now.hour,
            now.minute,
            now.second,
            self.r_string
        )

        return suffix

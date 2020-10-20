import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import pickle


class Sarsa_Agent:
    def __init__(self, states_n, actions_n, alpha=0.5, epsilon=0.1, gamma=0.95, epsilon_decay=True,
                 epsilon_decay_factor=0.01):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.states_n = states_n
        self.actions_n = actions_n
        self.Q = np.zeros((states_n, actions_n))
        self.new_a = None
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_factor = epsilon_decay_factor

    def act(self, state):
        # epsilon greedy
        if np.random.rand() < self.epsilon:
            act = np.random.choice(np.arange(self.actions_n))
        else:
            act = np.argmax(self.Q[int(state), :])
        return act

    def decay_epsilon(self, factor):
        self.epsilon -= factor if self.epsilon >= 0 else 0

    def update(self, new_s, r, s, a, done):
        self.new_a = self.act(new_s)
        mask = 0 if done else 1
        s, a, self.new_a, new_s = int(s), int(a), int(self.new_a), int(new_s)
        self.Q[s, a] += self.alpha * (r + self.gamma * self.Q[new_s, self.new_a] * mask - self.Q[s, a])
        if done and self.epsilon_decay:
            self.decay_epsilon(self.epsilon_decay_factor)
        return self.new_a

    def save(self, file_name="taxi.pkl"):
        with open(file_name, mode="wb") as f:
            pickle.dump(self.Q, f)

    def load(self, file_name="taxi.pkl"):
        with open(file_name, mode="rb") as f:
            self.Q = pickle.load(f)


class Deep_Agent:
    def __init__(self, state_size, action_size, env_name, gamma=0.99, alpha_w=0.15, alpha_theta=0.1):
        self.I = 1
        self.state_size = state_size
        self.action_size = action_size
        self.env_name = env_name
        self.gamma = gamma
        self.alpha_w = alpha_w
        self.alpha_theta = alpha_theta

        self.actor = keras.Sequential([keras.layers.Dense(50, activation="relu", input_shape=(self.state_size,)),
                                       keras.layers.Dense(action_size, activation=keras.activations.sigmoid)])

    def new_episode(self):
        self.I = 1

    def act(self, state):
        # Returns action from policy
        state = self.process_state(state)
        logits = self.actor(state)
        action_probs = tfp.distributions.Categorical(probs=logits)
        action = action_probs.sample()

        return action.numpy()[0]

    def process_state(self, state):
        return tf.convert_to_tensor(state.reshape(1, self.state_size), dtype=tf.float32)

    def load(self, src="brain_CartPole-v1.h5"):
        # self.actor = tf.keras.models.load_model(src)
        self.actor.load_weights(src)

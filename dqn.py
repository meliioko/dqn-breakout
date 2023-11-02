import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym
from tensorflow.keras import models, layers
import random

env = gym.make("Breakout-v0", obs_type='grayscale')
#env = gym.wrappers.RecordVideo(env, video_folder='videos/',name_prefix='atari' ,episode_trigger=lambda x: x % 200 == 0)



def build_model(action_size):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)))
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    return model



class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)



# Set parameters
N = 10000  # Replay memory capacity
M = 1000  # Number of episodes
T = 10000  # Max steps per episode
C = 1000  # Target network update frequency
action_size = env.action_space.n  # Number of actions
state_size = env.observation_space.shape[0]  # State size

# Initialize replay memory
D = deque(maxlen=N)


Q = build_model(action_size)
Q_hat = build_model(action_size)
Q_hat.set_weights(Q.get_weights())

# Preprocess function
def preprocess_state(state):
    return np.reshape(state, [1, state_size])

# Train
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.95  # Discount rate
M = 1

for episode in range(M):
    state = preprocess_state(env.reset())
    total_reward = 0
    for t in range(T):
        # Epsilon-greedy action selection
        if np.random.rand() <= epsilon:
            action = random.randrange(action_size)
        else:
            act_values = Q.predict(state)
            action = np.argmax(act_values[0])

        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(next_state)
        total_reward += reward

        # Store transition in D
        D.append((state, action, reward, next_state, done))

        state = next_state

        # Check if the episode is done
        if done:
            print(f"Episode: {episode}/{M}, Score: {total_reward}")
            break

        # Train using a random minibatch from D
        if len(D) > 32:
            minibatch = random.sample(D, 32)
            for w, a, r, w_next, terminal in minibatch:
                target = r
                if not terminal:
                    target = (r + gamma * np.amax(Q_hat.predict(w_next)[0]))
                target_f = Q.predict(w)
                target_f[0][a] = target
                Q.fit(w, target_f, epochs=1, verbose=0)

    # Update epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Update target network
    if episode % C == 0:
        Q_hat.set_weights(Q.get_weights())

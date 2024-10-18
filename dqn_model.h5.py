from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque
import logging

# Set up logging
logging.basicConfig(filename='cloud_mining_performance.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Simulate blockchain data (replace with actual data later)
def simulate_blockchain_data(num_samples=1000):
    return {
        'Block_Number': np.random.randint(1, 10000, num_samples),
        'Timestamp': np.random.randint(1, 10000000, num_samples),
        'Previous_Hash': np.random.randint(1, 1000000, num_samples),
        'Difficulty': np.random.randint(1, 20, num_samples),
        'Nonce': np.random.randint(1, 1000000, num_samples),
        'Transaction_Fees': np.random.uniform(0.01, 5.0, num_samples),
        'Block_Reward': np.random.uniform(6.25, 12.5, num_samples),
        'Valid_Block': np.random.randint(0, 2, num_samples)
    }

# BitcoinMiningEnv class (for simulating mining process)
class BitcoinMiningEnv:
    def __init__(self, difficulty_levels):
        self.difficulty_levels = difficulty_levels
        self.reset()

    def reset(self):
        self.state = random.choice(self.difficulty_levels)
        return self.state

    def step(self, action):
        valid_nonce = random.randint(1, 100) <= (100 / self.state)
        reward = 10 if valid_nonce else -1
        done = valid_nonce
        return self.state, reward, done

# DQNAgent class with model persistence
class DQNAgentWithPersistence:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename):
        self.model.save(filename)
        logging.info(f"Model saved as {filename}")

    def load_model(self, filename):
        self.model = load_model(filename)
        logging.info(f"Model loaded from {filename}")

# Simulate blockchain dataset
blockchain_data = simulate_blockchain_data()

# Use model persistence during training
def train_dqn_with_persistence(episodes=100, save_interval=10, model_filename='dqn_model.h5'):
    difficulty_levels = [5, 10, 15, 20]
    state_size = 1
    action_space = 100
    env = BitcoinMiningEnv(difficulty_levels)
    dqn_agent_with_persistence = DQNAgentWithPersistence(state_size=state_size, action_size=action_space)

    for episode in range(episodes):
        state = env.reset()
        state = np.reshape([state], [1, state_size])
        done = False
        total_reward = 0

        while not done:
            action = dqn_agent_with_persistence.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape([next_state], [1, state_size])
            dqn_agent_with_persistence.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if len(dqn_agent_with_persistence.memory) > 32:  # batch_size is 32
            dqn_agent_with_persistence.replay(32)

        if episode % save_interval == 0:
            dqn_agent_with_persistence.save_model(model_filename)

        logging.info(f"Episode {episode + 1} completed with total reward: {total_reward}")

    # Final save after training
    dqn_agent_with_persistence.save_model(model_filename)

# Train the model
train_dqn_with_persistence()

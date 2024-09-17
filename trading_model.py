import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#import streamlit as st

# Download historical data for a stock (e.g., Apple)
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data.head()



scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

# Train-test split
train_size = int(len(scaled_data) * 0.80)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

import tensorflow as tf
from tensorflow.keras import layers, models

# Generator model
def build_generator():
    model = models.Sequential([
        layers.Dense(256, input_dim=100, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(1024, activation="relu"),
        layers.Dense(1, activation="tanh")  # Output a single price value
    ])
    return model

# Discriminator model
def build_discriminator():
    model = models.Sequential([
        layers.Dense(1024, input_dim=1, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(1, activation="sigmoid")  # Binary classification
    ])
    return model


# Hyperparameters
latent_dim = 100

# Build generator and discriminator
generator = build_generator()
discriminator = build_discriminator()

# Compile discriminator
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Build and compile the GAN by combining generator and discriminator
discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
generated_price = generator(gan_input)
gan_output = discriminator(generated_price)

gan = models.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')

import matplotlib.pyplot as plt

def train_gan(epochs=1000, batch_size=128):
    for epoch in range(epochs):
        # Generate fake prices
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_prices = generator.predict(noise)

        # Sample real prices
        real_prices = train_data[np.random.randint(0, train_data.shape[0], batch_size)]

        # Labels for real and fake data
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        # Train discriminator
        d_loss_real = discriminator.train_on_batch(real_prices, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_prices, fake_labels)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real_labels)

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D Loss Real {d_loss_real[0]}, D Loss Fake {d_loss_fake[0]}, G Loss {g_loss}")
            
        # Optionally visualize the generated prices
        if epoch % 1000 == 0:
            plt.plot(generated_prices)
            plt.show()
# Train the GAN
train_gan(epochs=10000)

import gym
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float16)

    def reset(self):
        self.current_step = 0
        return self.data[self.current_step]

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = self.calculate_reward(action)
        return self.data[self.current_step], reward, done, {}

    def calculate_reward(self, action):
        # Simple reward logic (buy low, sell high)
        return 1 if action == 1 else 0

# Initialize the trading environment
env = TradingEnv(train_data)

from stable_baselines3 import PPO

# Define and train the agent
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Test the model on the test data
state = env.reset()
for _ in range(len(test_data)):
    action, _ = model.predict(state)
    state, reward, done, _ = env.step(action)
    if done:
        break

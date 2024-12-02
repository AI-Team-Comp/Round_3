import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import knu_rl_env.road_hog as road_hog

# Hyperparameters
GAMMA = 0.99          # Discount factor
LEARNING_RATE = 0.001 # Learning rate for the optimizer
BATCH_SIZE = 64       # Size of the minibatch for training
MEMORY_SIZE = 10000   # Replay memory size
EPSILON_START = 1.0   # Initial epsilon for ε-greedy policy
EPSILON_END = 0.01    # Minimum epsilon
EPSILON_DECAY = 0.995 # Decay rate for epsilon
TARGET_UPDATE_FREQ = 10 # Frequency to update target network
MAX_EPISODES = 500    # Maximum number of episodes to train

# Neural network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# Replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Process the observation into a single state vector
def process_state(observation, goal_spot, is_on_load, is_crashed, time):
    observation_vector = observation.flatten()
    return np.concatenate([observation_vector, goal_spot, [is_on_load, is_crashed, time]])

# Training loop
def train_dqn():
    env = road_hog.make_road_hog(show_screen=False)
    obs_space_size = 10 * 6 + 6 + 3  # observation (10x6), goal_spot (6), extra info (3)
    action_space_size = 9            # Number of actions

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(obs_space_size, action_space_size).to(device)
    target_net = DQN(obs_space_size, action_space_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)
    epsilon = EPSILON_START

    for episode in range(MAX_EPISODES):
        # Reset environment and take an initial step
        obs = env.reset()
        next_obs, _, terminated, truncated, _ = env.step(road_hog.RoadHogAgent.NON_ACCEL_NEUTRAL)  # Initial neutral action
        state = process_state(
            next_obs["observation"],
            next_obs["goal_spot"],
            next_obs["is_on_load"],
            next_obs["is_crashed"],
            next_obs["time"]
        )
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, action_space_size - 1)
            else:
                with torch.no_grad():
                    action = policy_net(state).argmax(dim=1).item()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = process_state(
                next_obs["observation"],
                next_obs["goal_spot"],
                next_obs["is_on_load"],
                next_obs["is_crashed"],
                next_obs["time"]
            )
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            if terminated or truncated:
                done = True

            # Store transition in memory
            memory.push((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            # Experience replay
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = list(zip(*transitions))

                state_batch = torch.cat(batch[0]).to(device)
                action_batch = torch.tensor(batch[1], dtype=torch.long).unsqueeze(1).to(device)
                reward_batch = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(device)
                next_state_batch = torch.cat(batch[3]).to(device)
                done_batch = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(device)

                # Q-learning update
                q_values = policy_net(state_batch).gather(1, action_batch)
                next_q_values = target_net(next_state_batch).max(1)[0].unsqueeze(1)
                expected_q_values = reward_batch + (1 - done_batch) * GAMMA * next_q_values

                loss = nn.MSELoss()(q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network
        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        print(f"Episode {episode + 1}/{MAX_EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    env.close()
    return policy_net

# Train the agent
trained_policy = train_dqn()


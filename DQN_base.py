import numpy as np
import torch
import torch.optim as optim
import random
from collections import deque
from knu_rl_env.road_hog import RoadHogAgent, make_road_hog, evaluate

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural network for DQN
class cDQN:
    def __init__(self, input_dim, output_dim):
        self.weights1 = torch.nn.Parameter(torch.randn(input_dim, 128, device=device) * 0.01)
        self.bias1 = torch.nn.Parameter(torch.zeros(128, device=device))
        
        self.weights2 = torch.nn.Parameter(torch.randn(128, 128, device=device) * 0.01)
        self.bias2 = torch.nn.Parameter(torch.zeros(128, device=device))
        
        self.weights3 = torch.nn.Parameter(torch.randn(128, output_dim, device=device) * 0.01)
        self.bias3 = torch.nn.Parameter(torch.zeros(output_dim, device=device))

    def forward(self, x):
        x = x.mm(self.weights1) + self.bias1
        x = relu(x)
        x = x.mm(self.weights2) + self.bias2
        x = relu(x)
        x = x.mm(self.weights3) + self.bias3
        return x

    def parameters(self):
        return [self.weights1, self.bias1, self.weights2, self.bias2, self.weights3, self.bias3]

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

# optimizer
class cAdam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize moments
        self.m = [torch.zeros_like(p, device=device) for p in self.params]
        self.v = [torch.zeros_like(p, device=device) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            grad = param.grad

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)

            # Compute bias-corrected moment estimates
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

class RoadHogRLAgent(RoadHogAgent):
    def __init__(self, policy_net):
        self.policy_net = policy_net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net.forward(state_tensor)
        action = q_values.argmax(dim=1).item()
        return action

# Process the observation into a single state vector
def process_state(observation, goal_spot, is_on_load, is_crashed, time):
    observation_vector = observation.flatten()
    return np.concatenate([observation_vector, goal_spot, [is_on_load, is_crashed, time]])

# 가중치 복사
def copy_weights(source, target):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)
        
# 손실함수
def relu(x):
    return torch.maximum(x, torch.zeros_like(x))

# Calculate reward about next observation
def calculate_reward(next_obs):
    reward = 0
    
    print(next_obs)

    # 도로 바깥으로 벗어나면 패널티
    if not next_obs["is_on_load"]:
        reward -= 10

    # 충돌 시 패널티
    if next_obs["is_crashed"]:
        reward -= 100

    # 거리 기반 보상
    distance_to_goal = np.linalg.norm([next_obs["observation"][0][0], next_obs["observation"][0][1]] - [next_obs["goal_spot"][0], next_obs["goal_spot"][1]])
    reward -= distance_to_goal * 0.1

    return reward

# Training loop
def train():
    env = make_road_hog(
        show_screen=True
    )
    
    obs_space_size = 10 * 6 + 6 + 3  # observation (10x6), goal_spot (6), extra info (3: is_on_load, is_crashed, time)
    action_space_size = 9            # Number of actions

    # Initialize networks
    policy_net = cDQN(obs_space_size, action_space_size)
    target_net = cDQN(obs_space_size, action_space_size)
    copy_weights(policy_net, target_net)  # Initial target network sync

    # Initialize optimizer
    optimizer = cAdam(policy_net.parameters(), lr=LEARNING_RATE)
    
    memory = ReplayMemory(MEMORY_SIZE)
    epsilon = EPSILON_START
    
    agent = RoadHogRLAgent(policy_net)

    for episode in range(MAX_EPISODES):
        obs = env.reset()   ## reset -> return only 2-dim array, pass
        next_obs, _, terminated, truncated, _ = env.step(RoadHogAgent.NON_ACCEL_NEUTRAL)
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
                    action = agent.policy_net.forward(state).argmax(dim=1).item()

            next_obs, _, terminated, truncated, _ = env.step(action)
            reward = calculate_reward(next_obs)
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
                q_values = agent.policy_net.forward(state_batch).gather(1, action_batch)
                next_q_values = target_net.forward(next_state_batch).max(1)[0].unsqueeze(1).detach()
                expected_q_values = reward_batch + (1 - done_batch) * GAMMA * next_q_values

                loss = ((q_values - expected_q_values) ** 2).mean()

                # Optimize model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network
        if episode % TARGET_UPDATE_FREQ == 0:
            copy_weights(agent.policy_net, target_net)

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Periodic logging
        if episode % 10 == 0:
            print(f"Episode {episode + 1}/{MAX_EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    env.close()
    return agent

if __name__ == '__main__':
    agent = train()
    evaluate(agent)
    # run_manual()
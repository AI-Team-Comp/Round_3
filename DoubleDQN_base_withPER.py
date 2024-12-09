import numpy as np
import torch
import random
import pickle
import time
from knu_rl_env.road_hog import RoadHogAgent, make_road_hog, evaluate

filename = "ddqn_test"

# Hyperparameters
GAMMA = 0.99          # Discount factor
LEARNING_RATE = 0.001 # Learning rate for the optimizer
BATCH_SIZE = 64       # Size of the minibatch for training
MEMORY_SIZE = 10000   # Replay memory size
EPSILON_START = 1.0   # Initial epsilon for ε-greedy policy
EPSILON_END = 0.01    # Minimum epsilon
EPSILON_DECAY = 0.995 # Decay rate for epsilon
TARGET_UPDATE_FREQ = 20 # Frequency to update target network
MAX_EPISODES = 300    # Maximum number of episodes to train

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

# Prioritized Experience Replay memory
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
    
    def push(self, transition, td_error):
        max_priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        
        self.priorities[self.position] = max_priority if td_error is None else (abs(td_error) + 1e-5)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.memory)]
        
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5

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
    def __init__(self, policy_net, action_space_size, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.policy_net = policy_net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.action_space_size = action_space_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def act(self, state, training=True):
        """
        Select an action based on ε-greedy policy.
        During training, ε-greedy is applied.
        During evaluation, the best action is chosen deterministically.
        """
        # If state is a dictionary, process it
        if isinstance(state, dict):
            state = self.process_state(
                state["observation"],
                state["goal_spot"],
                state["is_on_load"],
                state["is_crashed"],
                state["time"]
            )

        if training and random.random() < self.epsilon:
            # Random exploration
            action = random.randint(0, self.action_space_size - 1)
        else:
            # Exploitation (best action)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net.forward(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action
    
    # Process the observation into a single state vector
    def process_state(self, observation, goal_spot, is_on_load, is_crashed, time):
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
def calculate_ingame_reward(next_obs):
    reward = 0
    
    # print(next_obs)

    # 도로 바깥으로 벗어나면 패널티
    if not next_obs["is_on_load"]:
        reward -= 10

    # 충돌 시 패널티
    if next_obs["is_crashed"]:
        reward -= 50

    # 거리 기반 보상
    distance_to_goal = np.linalg.norm(
        np.array([next_obs["observation"][0][0], next_obs["observation"][0][1]]) - 
        np.array([next_obs["goal_spot"][0], next_obs["goal_spot"][1]])
    )
    reward -= distance_to_goal * 0.1

    return reward

def calculate_final_reward(obs):
    reward = 0
    time = obs["time"]
    distance_to_goal = np.linalg.norm(
        np.array([obs["observation"][0][0], obs["observation"][0][1]]) - 
        np.array([obs["goal_spot"][0], obs["goal_spot"][1]])
    )
    if distance_to_goal < 2:
        reward += 50000
        
    if time < 120:
        reward += 50000
    
    ## failure
    if time >= 120 and distance_to_goal > 2:
        reward -= 30000
        print("failure")
    return reward

# Training loop
def train():
    env = make_road_hog(show_screen=False)
    
    obs_space_size = 10 * 6 + 6 + 3  # observation (10x6), goal_spot (6), extra info (3: is_on_load, is_crashed, time)
    action_space_size = 9            # Number of actions

    # Initialize networks
    policy_net = cDQN(obs_space_size, action_space_size)
    target_net = cDQN(obs_space_size, action_space_size)
    copy_weights(policy_net, target_net)  # Initial target network sync

    # Initialize optimizer
    optimizer = cAdam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = PrioritizedReplayMemory(MEMORY_SIZE)
    
    losses = []

    agent = RoadHogRLAgent(policy_net, action_space_size)

    print("start!")
    start_time = time.time()
    for episode in range(MAX_EPISODES):
        obs = env.reset()
        next_obs, _, terminated, truncated, _ = env.step(RoadHogAgent.NON_ACCEL_NEUTRAL)

        # Process initial state using agent's process_state method
        state = agent.process_state(
            next_obs["observation"],
            next_obs["goal_spot"],
            next_obs["is_on_load"],
            next_obs["is_crashed"],
            next_obs["time"]
        )
        total_reward = 0
        done = False

        while not done:
            # Use ε-greedy policy from `act` method
            action = agent.act(state, training=True)

            next_obs, _, terminated, truncated, _ = env.step(action)
            reward = calculate_ingame_reward(next_obs)

            # Process next state using agent's process_state method
            next_state = agent.process_state(
                next_obs["observation"],
                next_obs["goal_spot"],
                next_obs["is_on_load"],
                next_obs["is_crashed"],
                next_obs["time"]
            )

            if terminated or truncated:
                done = True

            # Store transition in memory
            memory.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Experience replay
            if len(memory) >= BATCH_SIZE:
                transitions, indices, weights = memory.sample(BATCH_SIZE)
                batch = list(zip(*transitions))

                state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(device)
                action_batch = torch.tensor(np.array(batch[1]), dtype=torch.long).unsqueeze(1).to(device)
                reward_batch = torch.tensor(np.array(batch[2]), dtype=torch.float32).unsqueeze(1).to(device)
                next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(device)
                done_batch = torch.tensor(np.array(batch[4]), dtype=torch.float32).unsqueeze(1).to(device)

                # Q-learning update for Double DQN
                q_values = policy_net.forward(state_batch).gather(1, action_batch)
                next_actions = policy_net.forward(next_state_batch).argmax(1, keepdim=True)
                next_q_values = target_net.forward(next_state_batch).gather(1, next_actions).detach()
                expected_q_values = reward_batch + (1 - done_batch) * GAMMA * next_q_values

                td_errors = (q_values - expected_q_values).detach().cpu().numpy().squeeze()
                loss = (torch.tensor(weights, dtype=torch.float32).to(device) * (q_values - expected_q_values).pow(2)).mean()
                
                # tracking errors
                losses.append(loss.item())

                # Optimize model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                memory.update_priorities(indices, td_errors)
                
        if done:
            total_reward += calculate_final_reward(next_obs)
            # print(next_obs)
                
        # Update target network
        if episode % TARGET_UPDATE_FREQ == 0:
            copy_weights(agent.policy_net, target_net)

        # Decay epsilon
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

        if len(losses) > 0:
            avg_loss = np.mean(losses[-len(memory):])
        else:
            avg_loss = 0
        
        # Periodic logging
        if episode % 10 == 0:
            print(f"Episode {episode + 1}/{MAX_EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Avg Loss: {avg_loss:.4f}")
            
        if episode % 100 == 0:
            with open(f'{filename}_{episode}.pickle', 'wb') as fw:
                pickle.dump(agent, fw)

    print("done")
    end_time = time.time()
    
    env.close()
    
    elapsed_time = end_time - start_time
    print(f"학습에 걸린 시간: {elapsed_time:.2f}초")
    
    ## save agent
    with open(f'{filename}.pickle', 'wb') as fw:
        pickle.dump(agent, fw)
    
    return agent

def load(filename):
    try:
        with open(f'{filename}.pickle', 'rb') as fr:
            agent = pickle.load(fr)
        return agent
    except FileNotFoundError:
        print(f"Error: File '{filename}.pickle' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == '__main__':
    agent = train()
    # agent = load('dqn_test')
    evaluate(agent)
    # run_manual()
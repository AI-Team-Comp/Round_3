from knu_rl_env.road_hog import make_road_hog, evaluate
from DoubleDQN_base_withPER import *  ##RoadHogRLAgent, PrioritizedReplayMemory, HYPERPARAMETERS
import multiprocessing as mp
import numpy as np
import random
import time
import torch

OBS_SPACE_SIZE = 10 * 6 + 6 + 3  # observation (10x6), goal_spot (6), extra info (3)
ACTION_SPACE_SIZE = 9            # Number of actions

# Worker function for each agent
def work_agent(agent_id, policy_queue, experience_queue, env_fn, epsilon, max_steps=1000):
    np.random.seed(agent_id)  # Set unique seed for numpy
    random.seed(agent_id)     # Set unique seed for random

    env = env_fn()
    agent = RoadHogRLAgent(policy_queue, ACTION_SPACE_SIZE, epsilon)

    for episode in range(max_steps):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(obs)  # Choose action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Send experience to Policy Agent
            experience_queue.put((obs, action, reward, next_obs, done))
            obs = next_obs
            total_reward += reward

        if agent_id == 0:  # Optional logging for one agent
            print(f"Agent {agent_id}: Episode {episode}, Total Reward: {total_reward}")

    env.close()

# Policy agent to train the policy network
def policy_agent(policy_net, target_net, optimizer, memory, experience_queue, target_update_freq=20):
    steps = 0
    while True:
        # Collect experiences from queue
        while not experience_queue.empty():
            experience = experience_queue.get()
            memory.push(*experience)

        # Train policy if enough samples in memory
        if len(memory) >= BATCH_SIZE:
            transitions, indices, weights = memory.sample(BATCH_SIZE)
            batch = list(zip(*transitions))

            state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(device)
            action_batch = torch.tensor(np.array(batch[1]), dtype=torch.long).unsqueeze(1).to(device)
            reward_batch = torch.tensor(np.array(batch[2]), dtype=torch.float32).unsqueeze(1).to(device)
            next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(device)
            done_batch = torch.tensor(np.array(batch[4]), dtype=torch.float32).unsqueeze(1).to(device)

            # Double DQN Q-value updates
            q_values = policy_net.forward(state_batch).gather(1, action_batch)
            next_actions = policy_net.forward(next_state_batch).argmax(1, keepdim=True)
            next_q_values = target_net.forward(next_state_batch).gather(1, next_actions).detach()
            expected_q_values = reward_batch + (1 - done_batch) * GAMMA * next_q_values

            td_errors = (q_values - expected_q_values).detach().cpu().numpy().squeeze()
            loss = (torch.tensor(weights, dtype=torch.float32).to(device) * (q_values - expected_q_values).pow(2)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update priorities
            memory.update_priorities(indices, td_errors)

        # Update target network periodically
        if steps % target_update_freq == 0:
            copy_weights(policy_net, target_net)

        steps += 1

# Main function to run parallel training
def train_parallel():
    num_agents = 4  # Number of parallel agents
    env_fn = lambda: make_road_hog(show_screen=False)

    # Shared policy queue and experience queue
    policy_queue = mp.Queue()
    experience_queue = mp.Queue()

    # Initialize networks, optimizer, and memory
    policy_net = cDQN(OBS_SPACE_SIZE, ACTION_SPACE_SIZE)
    target_net = cDQN(OBS_SPACE_SIZE, ACTION_SPACE_SIZE)
    copy_weights(policy_net, target_net)
    optimizer = cAdam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = PrioritizedReplayMemory(MEMORY_SIZE)

    # Start work agents
    workers = []
    for agent_id in range(num_agents):
        epsilon = EPSILON_START * (EPSILON_DECAY ** agent_id)  # Decay epsilon for diversity
        worker = mp.Process(target=work_agent, args=(agent_id, policy_queue, experience_queue, env_fn, epsilon))
        worker.start()
        workers.append(worker)

    # Start policy agent
    policy_process = mp.Process(target=policy_agent, args=(policy_net, target_net, optimizer, memory, experience_queue))
    policy_process.start()

    # Join all processes
    for worker in workers:
        worker.join()
    policy_process.terminate()  # Terminate policy agent when workers are done

    # Save final model
    with open(f'{filename}.pickle', 'wb') as fw:
        pickle.dump(policy_net, fw)

    print("Training complete.")

if __name__ == '__main__':
    train_parallel()

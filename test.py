import pickle
from knu_rl_env.road_hog import evaluate, run_manual
from DQN_base import *

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
    agent = load('dqn_test2')
    evaluate(agent)
    # run_manual()
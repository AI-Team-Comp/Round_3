import numpy as np
import matplotlib.pyplot as plt
import gym

# key-value 쌍 형태로 값을 저장 가능
# key를 field명으로 값에 접근할 수 있어 편리
from collections import namedtuple

import random
import torch

from torch import nn
from torch import optim
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

ENV = 'CartPole-v1' # task
GAMMA = 0.99 # 시간할인율
MAX_STEPS = 200 # 한 episode당 최대 step 수
NUM_EPISODES = 500 # 최대 episode 수

BATCH_SIZE = 32
CAPACITY = 10000

class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY # Memeory에 최대 저장 건수
        self.memory = [] # 실제 transition을 저장할 Memory
        self.index = 0 # 저장 위치를 가리킬 index

    def push(self, state, action, state_next, reward):
        '''transition = (state, action, state_next, reward) Memory에 저장'''
        # 1. 메모리가 가득 차지 않은 경우
        if len(self.memory) < self.capacity:
            self.memory.append(None) 
        
        # 2. Transition 키-값 쌍의 형태로 값 저장
        self.memory[self.index] = Transition(state, action, state_next, reward)

        # push 했으니깐 다음 칸으로 index 옮기기.
        self.index = (self.index + 1) % self.capacity 
	
    def sample(self, batch_size):
        '''Memory에서 batch_size만큼 sampleing 하기'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''len 함수로 현재 저장된 transition 개수 return'''
        return len(self.memory)
    
# DQN 실제 수행
# Q함수를 딥러닝 신경망 형태로 정의

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions # 행동 수(왼쪽, 오른쪽)를 구함

        # transition을 기억하기 위한 Memory 객체 10000개 생성
        self.memory = ReplayMemory(CAPACITY)

        # Linear(4,32) -> ReLU() -> Linear(32,32) -> ReLU() -> Linear(32,2)
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states,32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32,32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))
        # print(self.model) # 신경망 구조 출력

        # 최적화 기법 선택
        self.optimizer = optim.Adam(self.model.parameters(), lr= 0.0001)

    
    def replay(self):
        '''Experience Replay로 신경망의 weight 학습'''

        # ------------------------
        # 1. 저장된 transition의 수가 미니배치보다 작으면 아무것도 하지 않기
        if len(self.memory) < BATCH_SIZE:
            return

        # ------------------------
        # 2. 미니 배치 생성
        # 2.1 ReplayMemory 객체의 sample 메소드로 미니 배치를 추출
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 transition를 미니 배치에 맞는 형태로 변형
        # transitions는 각 step 별로 (state, action, state_next, reward) 형태가 BATCH_SIZE만큼 저장됨.
        # (state, action, state_next, reward) * BATCH_SIZE --->
        # (state*BATCH_SIZE, action*BATCH_SIZE, state_next*BATCH_SIZE, reward*BATCH_SIZE) 형태로 변환
        # 예시 zip(*[(1,'hello'),(1,2)]) -> [(1,1),('hello',2)]
        batch = Transition(*zip(*transitions))

        # 2.3 state의 요소들을 미니 배치에 맞게 변형 후 신경망으로 다룰 수 있는 변수로 변형.
        # state를 예로 들면, [torch.FloatTensor of size 1*4] 형태의 요소가 BATCH_SIZE 개수만큼 있는 형태
        # 이를 torch.FloatTensor of size BATCH_SIZE*4 형태로 변형
        # state, action, reward, 최종이 아닌 state로 된 미니배치를 나타내는 변수 생성
        
        state_batch = torch.cat(batch.state)   # [BATCH_SIZE * 4]
        action_batch = torch.cat(batch.action) # [BATCH_SIZE * 1]
        reward_batch = torch.cat(batch.reward) # [BATCH_SIZE]
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        # ------------------------
        # 3. 정답 신호로 사용할 Q(s_t, a_t) 계산
        # 3.1 신경망 추론 모드로 전환
        self.model.eval()

        # 3.2 신경망으로 Q(s_t, a_t) 계산
        # self.model(state_batch)은 각 action에 대한 Q 값 출력
        # [BATCH_SIZE * 2] 형태. type은 FloatTensor
        # 여기서부터 실행한 행동 a_t에 대한 Q 값을 계산하므로 action_batch에서 취한 행동
        # action_batch에서 a_t가 0,1인지 index를 state별로 모아서 model의 output 값을 모으기.
        # axis=1 방향
        state_action_values = self.model(state_batch).gather(1, action_batch) # [BATCH_SIZE * 1]
        # self.model(state.batch)를 통과한 output 값을, action_batch의 action index에 맞춰서 선택함.
        

        # 3.3 max{(Q(s_t+1,a) 값 계산
        # 다음 state 존재 확인 필요. None 상태가 아니고 next_state 존재 확인하는 index 마스크 만들기
        # batch.next_state에 None에 따라서 tuple(map()) -> (False, False, False, True, False, ...)
        # (False, False, False, True, False, ...) dtype=torch.bool
        non_final_mask = torch.tensor(list(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool) # [BATCH_SIZE * 1]

        # 정답 신호 계산에 쓰일 next_state
        # 먼저 전체를 0으로 초기화
        next_state_values= torch.zeros(BATCH_SIZE) # [BATCH_SIZE]

        # state_next가 있는 index에 대한 최대 Q 값 구하기
        # model 출력 값에서 col 방향 최댓값(max(axis=1))이 되는 [value, index]를 구한다
        # 그리고 Q 값(index=0)을 출력한 다음
        # detach 메서드로 값 꺼내오기(학습과 독립적으로)
        #print("특이점",self.model(non_final_next_states).max(1)[0].detach())
        next_state_values[non_final_mask] = self.model(
            non_final_next_states).max(1)[0].detach()
        

        # 3.4 정답 신호로 사용할 Q(s_t, a_t) 값을 Q러닝 식으로 계산
        expected_state_action_values = reward_batch + GAMMA * next_state_values # [BATCH_SIZE]

        # 4. weight 수정
        # 4.1 신경망 학습모드
        self.model.train()

        # 4.2 손실함수를 계산(smooth_l1_loss는 Huber))
        # expected_state_action_values는 size가 [minibatch] -> unsqueez해서 [minibatch * 1]
        loss = F.smooth_l1_loss(state_action_values, 
                                expected_state_action_values.unsqueeze(1)) # axis=1에 새로운 차원 추가

        # 4.3 model 가중치 수정 (model(state_batch).gather)
        self.optimizer.zero_grad() # 경사 초기화
        loss.backward() # 역전파 계산
        self.optimizer.step() # 결합 가중치 수정

    # 2024
    def decide_action(self, state, episode):
        '''현재 state에 따라 actioon 결정''' 
        # e-greedy 알고리즘에서 서서히 최적 행동의 비중을 늘림
        epsilon = 0.5*(1/(episode+1))

        if epsilon <= np.random.uniform(0,1):
            self.model.eval() #신경망 추론 모드
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1,1)
        else:
            # 행동 무작위로 반환(0,1)
            action=torch.LongTensor(
                [[random.randrange(self.num_actions)]]) # 0 또는 1 중 행동을 무작위로 반환            # action은 [1*1] 형태 torch.LongTensor
        
        return action
        
class Agent:
    def __init__(self, num_states, num_actions):
        '''task의 state 및 action 수를 설정'''
        self.brain = Brain(num_states, num_actions) # Agent의 action을 결정할 Brain 객체 생성

    def update_q_function(self):
        '''Q 함수를 수정'''
        self.brain.replay()

    def get_action(self, state, episode):
        '''action 결정'''
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        ''' memory 객체에 state, action, state_next, reward 내용 저장'''
        self.brain.memory.push(state, action, state_next, reward) 

# CartPole을 실행헐 환경 정의
class Environment:
    def __init__(self):
        self.env = gym.make(ENV, render_mode='human') # task 설정
        num_states = self.env.observation_space.shape[0] # task 상태 변수 수 4
        num_actions = self.env.action_space.n # task action 수 2
        self.agent = Agent(num_states, num_actions) # agent 객체 생성

    def run(self):
        '''실행'''
        episode_10_list = np.zeros(10) # 최근 10 episode 동안 버틴 단계 수를 저장
                                       # (평균 step 수 출력)
        complete_episodes = 0 # 현재까지 195단계를 버틴 episode 수
        episode_final = False # 마지막 episode 여부

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()[0] # 환경 초기화

            state = observation # 관측을 변환 없이 그대로 state s로 사용\
            state = torch.from_numpy(state).type(torch.FloatTensor) # NumPy 변수 - Pytorch Tensor로 변환
            state = torch.unsqueeze(state, 0) # size 4를 size 1*4로 변환

            for step in range(MAX_STEPS): # 1 episode
                action = self.agent.get_action(state, episode) # 다음 행동 결정

                # 행동 a_t를 실행해 다음상태 s_{t+1}과 done 플래그 값 결정
                # action에 .item()을 호출해 행동 내용을 구함
                observation_next, _, done, _, _ = self.env.step(action.item()) # reward와 info는 사용하지 않음
                # 보상을 부여하고 episode의 종료 판정 및 state_next를 설정
                if done: # step > 200, 봉이 일정 각도 이상 기울면
                    state_next = None

                    # 최근 10 episdoe에서 버틴 step 수를 list에 저장
                    episode_10_list = np.hstack( (episode_10_list[1:], step+1) )

                    if step < 195:
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0
                    else:
                        reward = torch.FloatTensor([1.0])
                        complete_episodes = complete_episodes+1
                else:
                    reward = torch.FloatTensor([0.0])
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)

                # memory에 경험 저장
                self.agent.memorize(state, action, state_next, reward)

                # Experience Replay로 Q 함수 수정
                self.agent.update_q_function()

                # 관측 결과를 update
                state = state_next

                # episdoe 종료 처리
                if done:
                    print('%d Episode: Finished after %d steps: 최근 10 Episode의 평균 단계 수 = %.1lf' % (episode, step+1, episode_10_list.mean()))
                    break

                if episode_final is True:
                    # anmiation 생성 및 저장
                    break

                # 10 episode 연속으로 200단계 버티면 task 성공
                if complete_episodes >=10:
                    print('10 episode 연속 성공')
                    episode_final = True # 종료 생성
                    
cartpole_env = Environment()
cartpole_env.run()        
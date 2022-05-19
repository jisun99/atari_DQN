import copy
from collections import namedtuple
from itertools import count
import math    
import random   
import numpy as np
import time
import matplotlib.pyplot as plt
# with open('result_%s.csv' % (RESULT_POSTFIX), 'a') as fp:
# fp.write('idx,true,pred,same,dice04,dice05,dice06,dice07,dice08,path\n')

import gym   

from wrappers import *
from memory import ReplayMemory
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transion', ('state', 'action', 'reward', 'next_state', 'duration'))

def choose_action(state):
    global steps_done
    sample = random.random()

    eps_threshold = EP_END + (EP_START - EP_END) * math.exp(-1. * steps_done / EP_DECAY)
    steps_done += 1

    duration = 2
    time.sleep(duration)

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

def choose_duration(state, policy):
    global steps_done
    sample = policy(state.to('cuda')).max(1)[1].view(1, 1)
    
    duration_range = [1, 3, 5]
    duration = random.choice(duration_range)
    time.sleep(duration * 0.01)
    eps_threshold = EP_END + (EP_START - EP_END) * math.exp(-1. * steps_done / EP_DECAY)
    steps_done += 1
    
    if sample < eps_threshold:
        torch.tensor([[duration]], device=device, dtype=torch.long)
        return duration
    else:
        return 0

# batch sample train
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    
    batch = Transition(*zip(*transitions))

    # 1. action
    actions = tuple((map(lambda a: torch.tensor([[a]] ,device='cuda'), batch.action)))
    # 2. reward
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward)))
    # 3. duration
    durations = tuple((map(lambda d: torch.tensor([d], device='cuda'), batch.duration)))

    non_last_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.uint8)
    non_last_next_states = torch.cat([s for s in batch.next_state if s is not None]).to('cuda')

    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    duration_batch = torch.cat(durations)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_last_mask] = target_net(non_last_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch


    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))


    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    
# state
def get_state(obs):
    state = np.array(obs)
    # state = state[35:195, :, :]
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

# # train loop
# def train(env, n_episodes, render=False):
#     for episode in range(n_episodes):
#         obs = env.reset()
#         state = get_state(obs)
#         total_reward = 0.0
        
#         # until done  ~ count
#         for step in count():
    
#             action = choose_action(state)
#             duration = choose_duration(state)
#             if duration is not 0:
#                 print('total steps : {} \t duration: {}'.format(steps_done,duration))

#             if render:
#                 env.render()

#             obs, reward, done, info = env.step(action)
#             total_reward += reward

#             if not done:
#                 next_state = get_state(obs)
#             else:
#                 next_state = None

#             reward = torch.tensor([reward], device=device)
            
#             # push data in replay memory
     
#             memory.push(state, action.to('cuda'), reward.to('cuda'), next_state, duration)
#             state = next_state

#             # until not train model
#             if steps_done > INITIAL_MEMORY:
#                 optimize_model()
                
#                 # 1000's step -> target-net update
#                 if steps_done % TARGET_UPDATE == 0:
#                     target_net.load_state_dict(policy_net.state_dict())

#             if done:
#                 break
   
#         if episode % 1 == 0:
#             print('total steps: {} \t episodes: {}/{} \t total reward: {}'.format(steps_done, episode, step, total_reward))


#     print('model training is complete!!!\n')
#     env.close()
#     return

# train_duration
def train_duration(env, n_episodes, policy, render=False):
    scores = []
    mean_scores = []
    episodes = []
    mean_duration = []
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        
        # until done  ~ count
        for step in count():
            duration = choose_duration(state, policy)
            time.sleep(duration * 0.01)
            action = policy(state.to('cuda')).max(1)[1].view(1,1)

            #if duration is not 0:  
                #print('episode : {} \t total steps : {} \t duration: {}'.format(episode,steps_done,duration))

            if render:
                env.render()

            obs, reward, done, info = env.step(action)
            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)
            
            # push data in replay memory
     
            memory.push(state, action.to('cuda'), reward.to('cuda'), next_state, duration)
            state = next_state

            # until not train model
            #if steps_done > INITIAL_MEMORY:
                #optimize_model()
                
                # 1000's step -> target-net update
                #if steps_done % TARGET_UPDATE == 0:
                    #target_net.load_state_dict(policy_net.state_dict())

            if done:
                scores.append(total_reward)
                mean_scores.append(np.mean(scores))
                episodes.append(episode)
                mean_duration.append(np.mean(duration))
                break
   
        if episode % 1 == 0:
            print('total steps: {} \t episodes: {}/{} \t total reward: {}'.format(steps_done, episode, step, total_reward))


    print('model duration training is complete!!!\n')
    fig = plt.figure()
    ax = fig.subplots()
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Scores')
    ax.set_title('Atlantis')
    ax.plot(episodes, mean_scores, 'b')
    plt.savefig("./fig_save/0517_re1000ep_dqn_Atlantis_duration_score.png")
    plt.close(fig)
    
    fig2 = plt.figure()
    ax = fig2.subplots()
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Duration')
    ax.set_title('Atlantis')
    ax.plot(episodes, mean_duration, 'g')
    plt.savefig("./fig_save/0517_re1000ep_dqn_Atlantis_duration_duration.png")
    plt.close(fig2)
    
    env.close()
    print('Plotting also complete!!!\n')
    return

# trained-model -> test episode

def test(env, n_episodes, policy, render=False):

    # env = gym.wrappers.Monitor(env, './videos/' + 'dqn_breakoutNo_video')

    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for step in count():
            action= policy(state.to('cuda')).max(1)[1].view(1,1)
            duration = policy(state).max(1)[1].view(1, 1)

            if render:
                env.render()
                time.sleep(0.02)
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
     
            if not done:
                next_state = get_state(obs)
            else:
                next_state = None              
           

       
            reward = torch.tensor([reward], device=device)
            state = next_state
            
            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break
    env.close()
    return

if __name__=='__main__':
    device = torch.device("cuda")

    BATCH_SIZE = 16             
    GAMMA = 0.9                
    EP_START = 1               
    EP_END = 0.02               

    EP_DECAY = 10000          
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4                 
    INITIAL_MEMORY = 100000
    MEMORY_SIZE = 10 * INITIAL_MEMORY
    
    # game action space -> n_actions
    policy_net = DQN(n_actions=4).to(device)
    target_net = DQN(n_actions=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0

    env = gym.make('AtlantisNoFrameskip-v4')
    env = make_env(env)

    memory = ReplayMemory(MEMORY_SIZE)

    # episode
    # train(env, 1000)

    #torch.save(policy_net, "0513_duration_dqn_breakoutNo_model")
  
    policy_net = torch.load("1000dqn_Atlantis_model")
    train_duration(env, 1000, policy_net, render=False)
    torch.save(policy_net, "220520_10ep_duration_dqn_Atlantis_model")
    #test(env, 1, policy_net, render=False)


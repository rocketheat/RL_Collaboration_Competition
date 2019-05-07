from unityagents import UnityEnvironment
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
from collections import namedtuple, deque

from time import sleep


# select this option to load version 1 (with a single agent) of the environment
env = UnityEnvironment(file_name='./Tennis.app')


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# environment information
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
n_agents = len(env_info.agents)
print('Number of agents:', n_agents)

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like: ', states[0])


from train import Actor, Critic, ReplayBuffer, OUNoise

DEVICE = 'cpu'

# hyperparameters
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 2e-1              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

ADD_NOISE = True
SEED = 2

N_EPISODES = 15

CHECKPOINT_FOLDER = './Saved_Model/'

shared_memory = ReplayBuffer(DEVICE, action_size, BUFFER_SIZE, BATCH_SIZE, SEED)
noise = OUNoise(action_size, 2)

ACTOR_1_KEY = 0
ACTOR_2_KEY = 1

actor_1 = Actor(DEVICE, ACTOR_1_KEY, state_size, action_size, SEED, shared_memory, noise, LR_ACTOR, WEIGHT_DECAY, CHECKPOINT_FOLDER)
actor_2 = Actor(DEVICE, ACTOR_2_KEY, state_size, action_size, SEED, shared_memory, noise, LR_ACTOR, WEIGHT_DECAY, CHECKPOINT_FOLDER)
critic = Critic(DEVICE, state_size, action_size, SEED, GAMMA, TAU, LR_CRITIC, WEIGHT_DECAY, CHECKPOINT_FOLDER)

def maddpg_train(n_episodes=N_EPISODES, train=True):
    scores = []
    scores_window = deque(maxlen=100)
    average_scores_list = []

    for episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]            # reset the environment
        states = env_info.vector_observations                        # get initial states
        actor_1.reset()                                              # reset the agent noise
        actor_2.reset()                                              # reset the agent noise

        score = np.zeros(n_agents)

        while True:
            action_1 = actor_1.act( states[ACTOR_1_KEY], ADD_NOISE )
            action_2 = actor_2.act( states[ACTOR_2_KEY], ADD_NOISE )
            actions = np.concatenate( (action_1, action_2) )

            env_info = env.step( actions  )[brain_name]              # send the action to the environment
            next_states = env_info.vector_observations               # get the next state
            rewards = env_info.rewards                               # get the reward
            dones = env_info.local_done                              # see if episode has finished

            if train:

                actor_1.step(states[ACTOR_1_KEY], action_1, rewards[ACTOR_1_KEY], next_states[ACTOR_1_KEY], dones[ACTOR_1_KEY])
                actor_2.step(states[ACTOR_2_KEY], action_2, rewards[ACTOR_2_KEY], next_states[ACTOR_2_KEY], dones[ACTOR_2_KEY])

                critic.step(actor_1, shared_memory)
                critic.step(actor_2, shared_memory)

            score += rewards                                         # update the score

            states = next_states

            sleep(0.01) #Slow environement for visualization                                   # roll over the state to next time step

            if np.any( dones ):                                      # exit loop if episode finished
                break

        scores.append(np.mean(score))
        scores_window.append(np.mean(score))

        print('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f}'.format(episode, np.mean(score), np.mean(scores_window)), end="")

    return scores


# train the agent
scores = maddpg_train(train=False)

env.close()

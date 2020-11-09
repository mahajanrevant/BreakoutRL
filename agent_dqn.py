#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
import math
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        
        #Gym parameters
        self.num_actions = env.action_space.n
        
        # parameters for repaly buffer
        self.buffer_max_len = 10000
        self.buffer = deque(maxlen=self.buffer_max_len)

        # paramters for neural network
        self.batch_size = 32
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.target_decay = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #Training
        self.max_steps = 50
        self.steps_done = 0
        self.num_episode = 1
        self.target_update = 10
        self.learning_rate = 1.5e-4
        
        # Neural Network
        self.policy_net = DQN()
        self.target_net = DQN()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            # Use this to load a model
            # self.policy_net = torch.load()

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1 * self.steps_done / self.eps_decay)
        observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0).permute(0,3,1,2)
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(observation).max(1)[1].item()
        else:
            return self.env.action_space.sample()
        ###########################
    
    def push(self, state, reward, action, next_state, done):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.buffer.append((state, reward, action, next_state, done))
        ###########################
        
        
    def replay_buffer(self, batch_size):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        batch = random.sample(self.buffer, batch_size)
        states = []
        rewards = []
        actions = []
        next_states = []
        dones = []
        for sample in batch:
            state, reward, action, next_state, done = sample
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            next_states.append(next_state)
            dones.append(done)
        ###########################
        return states, rewards, actions, next_states, dones

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        loss = self.compute_loss(self.replay_buffer(self.batch_size))
        self.optimizer.zero_grad()
        loss.backwards()
        self.optimizer.step()

    def compute_loss(self, states, rewards, actions, next_states, dones):
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.int).to(self.device)

        Q_current = self.policy_net.forward(states).gather(1, actions.unsqueeze(1))

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        print("I am here")
        for episode in range(self.num_episode):
            observation = self.env.reset()
            done = False

            ## Not sure if this is the right way to do this?
            for step in range(self.max_steps):
                action = self.make_action(observation)
                new_observation, reward, done, _ = env.step(action)
                self.push(observation, reward, action, new_observation, done)

                ## Updating the network
                self.update()

                ##
                if done:
                    break

                observation = new_observation

                if episode % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
        ###########################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
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

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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
        self.model = None
        self.batch_size = 32
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.target_decay = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Neural Network
        self.policy_net = DQN()
        self.target_net = DQN()
        
        #Training
        self.max_steps = 50
        self.steps_done = 0
        self.num_episode = 50
        self.target_update = 10


        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            self.policy_net = torch.load()

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
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(observation).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]],
             device=self.device, dtype=torch.long)
        ###########################
    
    def push(self, state, reward, action, done):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.buffer.append((state, reward, action, done))
        ###########################
        
        
    def replay_buffer(self, batch_size):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        batch = random.sample(self.buffer, batch_size)
        ###########################
        return batch

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = self.replay_buffer()

        

    def compute_loss(self, batch):
        pass    

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        episode_rewards = []
        for episode in range(self.num_episode):
            observation = self.env.reset()
            done = False

            ## Not sure if this is the right way to do this?
            for step in range(self.max_steps):
                action = self.make_action(observation)
                new_observation, reward, done, _ = env.step(action)
                self.push(observation, reward, action, done)

                ## Updating the network
                self.update()

                ##
                if done or step == self.max_steps - 1:
                    break

                observation = new_observation

                if episode % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
        ###########################

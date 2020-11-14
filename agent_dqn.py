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
        self.buffer_max_len = 20000
        self.buffer = deque(maxlen=self.buffer_max_len)
        self.episode_reward_list = []
        self.moving_reward_avg = []

        # paramters for neural network
        self.batch_size = 32
        self.gamma = 0.999
        self.eps_threshold = 0
        self.eps_start = 1
        self.eps_end = 0.025
        self.max_expisode_decay = 10000
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #Training
        self.steps_done = 0
        self.num_episode = 20000
        self.target_update = 5000
        self.learning_rate = 1.5e-4
        
        # Neural Network
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            self.policy_net.load_state_dict(torch.load("test_model.pt"))
            self.policy_net.eval()
            ###########################
            # YOUR IMPLEMENTATION HERE #
    
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
        with torch.no_grad():
            sample = random.random()

            ## Check if this is the best way to decline
            observation = torch.tensor(observation, dtype=torch.float, device=self.device).permute(2,0,1).unsqueeze(0)

            if test:
                return self.policy_net(observation).max(1)[1].item()

            if sample > self.eps_threshold:
                #print("Above threshold")
                    return self.policy_net(observation).max(1)[1].item()
            else:
                #print("Below Threshold")
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
        if self.steps_done < 5000:
            return
        states, rewards, actions, next_states, dones = self.replay_buffer(self.batch_size)
        loss = self.compute_loss(states, rewards, actions, next_states, dones)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp(-1,1)
        self.optimizer.step()

    def compute_loss(self, states, rewards, actions, next_states, dones):
        non_final_mask = [not done for done in dones]
             
        states = torch.tensor(states, dtype=torch.float).permute(0,3,1,2).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).permute(0,3,1,2).to(self.device)
        dones = torch.tensor(dones, dtype=torch.long).to(self.device)
        
        Q_current = self.policy_net.forward(states).gather(1, actions.unsqueeze(1))
        Q_current = Q_current.squeeze(1)
        ## Should do this with no grad

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(next_states[non_final_mask]).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + rewards
        
        loss = F.smooth_l1_loss(Q_current, expected_state_action_values)
        
        del states, rewards, actions, next_states, dones, Q_current, next_state_values, expected_state_action_values
        
        return loss
        
    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        for episode in range(self.num_episode):
            #Check this please
            observation = self.env.reset() / 255
            
            self.eps_threshold = max(1 + (((self.eps_end - self.eps_start)/self.max_expisode_decay) * episode),
                                     self.eps_end)
            episode_steps = 0
            done = False
            episode_reward = 0
            ## Not sure if this is the right way to do this?
            while not done:
                action = self.make_action(observation, test=False)
                new_observation, reward, done, _ = self.env.step(action)
                
                new_observation = new_observation / 255
                episode_reward += reward
                self.steps_done += 1
                episode_steps += 1
                
                self.push(observation, reward, action, new_observation, done)
                
                ## Updating the network
                self.update()

                observation = new_observation

                if self.steps_done % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            self.episode_reward_list.append(episode_reward)
          
            if episode % 100 == 0:
                print('episode: {} reward: {} episode length: {}'.format(episode,
                                                                        episode_reward,
                                                                        episode_steps))
                torch.save(self.policy_net.state_dict(), 'test_model.pt')
        ###########################
        print("Done")
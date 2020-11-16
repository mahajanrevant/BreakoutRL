#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
import math
from collections import deque
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
from dueling_dqn_model import DuelingDQN

from environment import Environment
from agent import Agent
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
        
        #Buffer
        self.buffer_max_len = 20000
        self.buffer = deque(maxlen=self.buffer_max_len)
        
        #Training Parameters
        self.num_episodes = 30000
        self.batch_size = 32
        self.learning_rate = 1.5e-4
        self.steps_done = 0
        self.target_update = 5000
        self.step_start_learning = 5000
        self.gamma = 0.999
        self.epsilon_start = 1
        self.epsilon_end = 0.025
        self.epsilon_decay_steps = 100000
        self.epsilon = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        #TODO: Print this out and see this
        self.delta_epsilon = (self.epsilon_start - self.epsilon_end)/self.epsilon_decay_steps
            
        #Model
        self.policy_net = DuelingDQN().to(self.device)
        self.target_net = DuelingDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        
        #Values to be printed
        self.episode_reward_list = []
        self.moving_reward_avg = []
        
        if args is not None and args.test_dqn:
            #you can load your model here
            print('loading trained model')
            
            self.policy_net = DuelingDQN()
            self.policy_net.load_state_dict(torch.load("test_model.pt", map_location=torch.device('cpu')))
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
            observation_np = np.array(observation, dtype=np.float32)
            observation_tensor = torch.as_tensor(np.expand_dims(observation_np / 255., axis=0),
                                            device=self.device).permute(0,3,1,2)
            #TODO: Change this to test_net?
            if test:
                return self.policy_net(observation_tensor).max(1)[1].item()
            
            result = np.random.uniform()         
            
            ## Possibly put this on CPU?
            if result > self.epsilon:
                return self.policy_net(observation_tensor).max(1)[1].item()
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
        states = torch.as_tensor(np.array(states), device=self.device)
        actions = torch.as_tensor(np.array(actions), device=self.device)
        rewards = torch.as_tensor(np.array(rewards, dtype=np.float32),
                                  device=self.device)
        next_states = torch.as_tensor(np.array(next_states), device=self.device)
        dones = torch.as_tensor(np.array(dones, dtype=np.float32),
                                device=self.device)
        ###########################
        return states, rewards, actions, next_states, dones

    def update(self):
        
        if len(self.buffer) < self.step_start_learning:
            return     
        
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.delta_epsilon
        
        states, rewards, actions, next_states, dones = self.replay_buffer(self.batch_size)
        states = states.type(torch.FloatTensor).to(self.device).permute(0,3,1,2) / 255.
        next_states = next_states.type(torch.FloatTensor).to(self.device).permute(0,3,1,2) / 255.        
        loss = self.compute_loss(states, rewards, actions, next_states, dones)
        self.optimizer.zero_grad()
        loss.backward()
        # Check this
        for param in self.policy_net.parameters():
            param.grad.data.clamp(-1,1)
        self.optimizer.step()
        return loss

    def compute_loss(self, states, rewards, actions, next_states, dones):
        non_final_mask = [not done for done in dones]
            
        Q_current = self.policy_net.forward(states).gather(1, actions.unsqueeze(1))
        Q_current = Q_current.squeeze(1)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(next_states[non_final_mask]).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + rewards
        
        ## Check if they come to the same thing after converging
        loss = self.loss_fn(Q_current, expected_state_action_values.detach())
        
        del Q_current, next_state_values, expected_state_action_values
        
        return loss
        
    def test(self): 
        test_env = Environment('BreakoutNoFrameskip-v4', None, atari_wrapper=True, test=True)
        agent = Agent_DQN(test_env, None)
        rewards = []
        seed = 11037
        total_episodes=30
        test_env.seed(seed)
        for i in range(total_episodes):
            state = test_env.reset()
            done = False
            episode_reward = 0.0

            #playing one game
            while(not done):
                action = agent.make_action(state, test=True)
                state, reward, done, info = test_env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)
        print('Run %d episodes'%(total_episodes))
        print('Mean:', np.mean(rewards))
    
    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        for episode in range(self.num_episodes):
            
            observation = self.env.reset()
            episode_steps = 0
            episode_reward = 0
            done = False
            
            while not done:

                action = self.make_action(observation, test=False)
                new_observation, reward, done, _ = self.env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                self.steps_done += 1
                
                self.push(observation, reward, action, new_observation, done)
                observation = new_observation

                self.update()
                
                if self.steps_done % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            
            self.episode_reward_list.append(episode_reward)
            self.moving_reward_avg.append(np.average(np.array(self.episode_reward_list[-30:])))

            if episode % 25 == 0:
                print('episode: {} epsilon: {} average reward: {} episode length: {}'.format(episode,
                                                                        self.epsilon,
                                                                        self.moving_reward_avg[-1],
                                                                        episode_steps))
                torch.save(self.policy_net.state_dict(), 'test_model.pt')
                np_moving_reward_avg = np.array(self.moving_reward_avg)
                np.savetxt("rewards.csv", np_moving_reward_avg, delimiter=",") 
            
            if episode % 500 == 0:
                self.test()
          
        self.moving_reward_avg = np.array(self.moving_reward_avg)
        np.savetxt("rewards.csv", self.moving_reward_avg, delimiter=",")            
        print("Done")
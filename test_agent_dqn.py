import argparse
#from test import test
from environment import Environment
from agent_dqn import Agent_DQN
from main import parse
from dqn_model import DQN
import torch



if __name__ == '__main__':
    args = parse()
    env_name = 'BreakoutNoFrameskip-v4'
    env = Environment(env_name, args, atari_wrapper=True)
    agent = Agent_DQN(env, args)
    print (agent.buffer)
    for i in range(0, 8):
        agent.push(i)
    print(agent.buffer)
    dqn = DQN()
    print(dqn)
    input = torch.randn(1, 4, 84, 84)
    out = dqn(input)
    print(out.size())

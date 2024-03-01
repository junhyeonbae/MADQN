import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv
import numpy as np
from collections import deque
import torch.optim as optim
from arguments import args
import random


class G_DQN_1(nn.Module):
    def __init__(self,  dim_act, observation_state):
        super(G_DQN_1, self).__init__()
        self.eps_decay = args.eps_decay

        self.observation_state = observation_state
        self.dim_act = dim_act
        self.tanh = nn.Tanh()

        #GRAPH
        self.dim_feature = observation_state[2]
        self.gnn1 = DenseSAGEConv(self.dim_feature, 128)
        self.gnn2 = DenseSAGEConv(128, 32)
        self.sig = nn.Sigmoid() #sigmoid ? ??? ?? ??..

        #DQN
        self.dim_input = ((args.predator1_view_range *2)**2)*32
        self.FC1 = nn.Linear(self.dim_input, 128)
        self.FC2 = nn.Linear(128, dim_act)

        self.relu = nn.ReLU()


    def forward(self, x, adj):

        try:
            torch.cuda.empty_cache()
        except:
            pass


        if isinstance(x, np.ndarray):
            x = torch.tensor(x).float()
        else:
            pass

        x_pre = x.reshape(-1, self.dim_feature)

        x = self.gnn1(x_pre, adj)
        x = self.tanh(x)
        print("x.shape",x.shape)
        x = self.gnn2(x, adj).squeeze()
        x = self.tanh(x)
        print("x.shape", x.shape)
        x = torch.flatten(x)
        print("x.shape", x.shape)

        x = self.FC1(x)
        x = self.tanh(x)
        x = self.FC2(x)
        print("x.shape", x.shape)

        return x

class G_DQN_2(nn.Module):
    def __init__(self,  dim_act, observation_state):
        super(G_DQN_2, self).__init__()
        self.eps_decay = args.eps_decay

        self.observation_state = observation_state
        self.dim_act = dim_act
        self.tanh = nn.Tanh()

        #GRAPH
        self.dim_feature = observation_state[2]
        self.gnn1 = DenseSAGEConv(self.dim_feature, 128)
        self.gnn2 = DenseSAGEConv(128, 32)
        self.sig = nn.Sigmoid() #sigmoid ? ??? ?? ??..

        #DQN
        self.dim_input = ((args.predator2_view_range *2)**2)*32
        self.FC1 = nn.Linear(self.dim_input, 128)
        self.FC2 = nn.Linear(128, dim_act)

        self.relu = nn.ReLU()


    def forward(self, x, adj):

        try:
            torch.cuda.empty_cache()
        except:
            pass


        if isinstance(x, np.ndarray):
            x = torch.tensor(x).float()
        else:
            pass

        x_pre = x.reshape(-1, self.dim_feature)

        x = self.gnn1(x_pre, adj)
        x = self.tanh(x)
        print("x.shape",x.shape)
        x = self.gnn2(x, adj).squeeze()
        x = self.tanh(x)
        print("x.shape", x.shape)
        x = torch.flatten(x)
        print("x.shape", x.shape)

        x = self.FC1(x)
        x = self.tanh(x)
        x = self.FC2(x)

        return x


class ReplayBuffer:
   def __init__(self, capacity=args.buffer_size):
      self.buffer = deque(maxlen=capacity)

   def put(self, observation , action , reward, next_observation, termination, truncation):
       self.buffer.append([observation, action , reward, next_observation, termination, truncation])



   def sample(self):
       sample = random.sample(self.buffer, 1)

       observation, action , reward, next_observation, termination, truncation = zip(*sample)
       return observation, action , reward, next_observation, termination, truncation

   def size(self):
      return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, dim_act, observation_state):
        super(DQN, self).__init__()
        self.observation_state = observation_state
        self.dim_act = dim_act

        # DQN
        self.dim_input = np.prod(observation_state)
        self.FC1 = nn.Linear(self.dim_input, 128)
        self.FC2 = nn.Linear(128, dim_act)
        self.relu = nn.ReLU()

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float()
        else:
            pass

        x = state.reshape(-1, self.dim_input)
        x = self.relu(self.FC1(x))
        x = self.FC2(x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv
import numpy as np
from collections import deque
import torch.optim as optim
from arguments import args
import random


class G_DQN(nn.Module):
    def __init__(self,  dim_act, observation_state):
        super(G_DQN, self).__init__()
        self.eps_decay = args.eps_decay

        self.observation_state = observation_state
        self.dim_act = dim_act
        self.tanh = nn.Tanh()

        #GRAPH
        self.dim_feature = observation_state[2]
        self.gnn1 = DenseSAGEConv(self.dim_feature, 128)
        self.gnn2 = DenseSAGEConv(128, self.dim_feature*2)
        self.sig = nn.Sigmoid() #sigmoid ? ??? ?? ??..

        #info
        self.dim_info = observation_state[2]
        self.gnn1_info = DenseSAGEConv(self.dim_feature,  128)
        self.gnn2_info = DenseSAGEConv(128, self.dim_feature)

        #DQN
        self.dim_input = observation_state[0] * observation_state[1] * observation_state[2]*2
        self.FC1 = nn.Linear(self.dim_input, 128)
        self.FC2 = nn.Linear(128, dim_act)

        self.relu = nn.ReLU()


    def forward(self, x, adj, info):

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
        x = self.tanh(x[0])
        x = self.gnn2(x, adj).squeeze()
        x = self.tanh(x)

        dqn = x[:, :self.dim_feature]

        shared = self.tanh(x[:, self.dim_feature:])

        shared = dqn * shared
        shared = shared.reshape(self.observation_state)
        #dqn = dqn.reshape(self.observation_state)

        info_pre = info.reshape(-1, self.dim_feature)
        x1 = self.gnn1_info(info_pre,adj)
        x1 = self.tanh(x1)
        x1 = self.gnn2_info(x1[0], adj).squeeze()
        x1 = self.tanh(x1)
        x1 = x1.reshape(self.observation_state)


        x = torch.cat((shared, x1), dim=0).reshape(-1, self.dim_input).squeeze()

        x = self.FC1(x)
        x = self.tanh(x)
        x = self.FC2(x)


        return x, shared.detach()



class ReplayBuffer:
   def __init__(self, capacity=args.buffer_size):
      self.buffer = deque(maxlen=capacity)

   def put(self, observation, book , action , reward, next_observation, book_next, termination, truncation):
       self.buffer.append([observation, book, action , reward, next_observation, book_next, termination, truncation])



   def sample(self):
       sample = random.sample(self.buffer, 1)

       observation, book , action , reward, next_observation, book_next, termination, truncation = zip(*sample)
       return observation, book , action , reward, next_observation, book_next, termination, truncation

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


class ReplayBuffer_cen:
   def __init__(self, capacity=args.buffer_size):
      self.buffer = deque(maxlen=capacity)

   def put(self, observation , action , reward, next_observation, termination, truncation):
       self.buffer.append([observation, action , reward, next_observation, termination, truncation])


   def sample(self):
       sample = random.sample(self.buffer, 1)

       observation , action , reward, next_observation, termination, truncation = zip(*sample)
       return observation , action , reward, next_observation, termination, truncation

   def size(self):
      return len(self.buffer)
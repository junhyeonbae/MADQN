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

        #GRAPH
        self.dim_feature = observation_state[2]
        self.gnn1 = DenseSAGEConv(self.dim_feature, 128)
        self.gnn2 = DenseSAGEConv(128, self.dim_feature*2) #feature ? 2?? ??
        self.sig = nn.Sigmoid() #sigmoid ? ??? ?? ??..

        #DQN
        self.dim_input = observation_state[0] * observation_state[1] * observation_state[2]*2 #concat ?? 2?!
        self.FC1 = nn.Linear(self.dim_input, 128)
        self.FC2 = nn.Linear(128, dim_act)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()


    def forward(self, x, adj, info): #x? adj? ??? ???? ??  GSAGE? ??? ???? ??? ??, from_guestbook ?? ??? ?? (8*8*7)? ??? ????

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
        #x = F.elu(self.gnn2(x,adj)).squeeze()  # exponential linear unit #squeeze ? ?? ??: x? batch_size? ???? ?? ? ??? ?? ?? ??? 1*100*3?? ??->100*3?? ???? ??
        x = self.gnn2(x, adj).squeeze()

        #x = F.elu(self.gnn2(self.gnn1(x.reshape(-1, self.dim_feature) ,adj),adj)).squeeze()

        dqn = x[:, :self.dim_feature]  #   100*3 : ?? x? ??? dqn ?? ???? ??? ??? sigmoid??? ??? ?? ????? ??.

        shared = self.sig(x[:, self.dim_feature:]) #share graph ? ??? ?! ????
        shared = dqn * shared # sigmoid ?? ?? x? dot???
        shared = shared.reshape(self.observation_state) #?? 10*10*5 ?? ?????? ?-> ?? shared graph ? ????? ??.
        #dqn = dqn.reshape(self.observation_state)


        x = torch.cat((shared, info), dim=0).reshape(-1, self.dim_input)
        #print("x : shape",x.shape)
        x = self.FC1(x)
        #print("x : shape", x.shape)
        x = self.FC2(x)
        #print("x : shape", x.shape)

        return x, shared.detach()




class ReplayBuffer:
   def __init__(self, capacity=args.buffer_size):
      self.buffer = deque(maxlen=capacity)

   def put(self, observation, book , action , reward, next_observation, book_next, termination, truncation):
       self.buffer.append([observation, book, action , reward, next_observation, book_next, termination, truncation]) #[state, action, reward, next_state, done]??? ??? history? ??



   def sample(self):
       sample = random.sample(self.buffer, 1)  # batch size?? buffer?? ????.

       observation, book , action , reward, next_observation, book_next, termination, truncation = zip(*sample)
       return observation, book , action , reward, next_observation, book_next, termination, truncation  # buffer?? ??? ??? ???? ??? ??

   def size(self):
      return len(self.buffer)   #buffer ??????? ?? ?


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
       self.buffer.append([observation, action , reward, next_observation, termination, truncation]) #[state, action, reward, next_state, done]??? ??? history? ??

#

   def sample(self):
       sample = random.sample(self.buffer, 1)  # batch size?? buffer?? ????.

       observation , action , reward, next_observation, termination, truncation = zip(*sample)
       return observation , action , reward, next_observation, termination, truncation  # buffer?? ??? ??? ???? ??? ??

   def size(self):
      return len(self.buffer)   #buffer ??????? ?? ?
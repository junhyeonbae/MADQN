from model_decen import G_DQN_1,G_DQN_2, ReplayBuffer
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from arguments import args


predator1_view_range = args.predator1_view_range
predator2_view_range = args.predator2_view_range
n_predator1 = args.n_predator1
n_predator2 = args.n_predator2
n_prey = args.n_prey


shared_shape = (args.map_size + (predator1_view_range-2)*2, args.map_size + (predator1_view_range-2)*2, 3)
predator1_obs = (predator1_view_range*2,predator1_view_range*2,3)
predator2_obs = (predator2_view_range*2,predator2_view_range*2,3)

dim_act = 13

def king_adj(n) :

    A = np.zeros((n**2, n**2))

    for i in range(n**2):

        if i // n == 0 :

            if i % n == 0 :

                A[i, i+1] = 1
                A[i, i+n] = 1
                A[i, i+1+n] = 1

            elif i % n == n-1:

                A[i, i-1] = 1
                A[i, i-1+n] = 1
                A[i, i+n] = 1

            else:
                A[i, i-1] = 1
                A[i, i+1] = 1
                A[i, i-1+n] = 1
                A[i, i+n] = 1
                A[i, i+1+n] = 1

        elif i // n == n-1:

            if i % n == 0:

                A[i, i-n] = 1
                A[i, i+1-n] = 1
                A[i, i+1] = 1
            elif i % n == n-1:

                A[i, i-n] = 1
                A[i, i-1-n] = 1
                A[i, i-1] = 1
            else:
                A[i, i - 1] = 1
                A[i, i + 1] = 1
                A[i, i - 1-n] = 1
                A[i, i-n] = 1
                A[i, i+ 1-n] = 1

        else:
                if i % n == 0:
                    A[i, i-n] = 1
                    A[i, i+1-n] = 1
                    A[i, i+1] = 1
                    A[i, i+n] = 1
                    A[i, i+1+n] = 1

                elif i % n == n-1:
                    A[i, i-1-n] = 1
                    A[i, i-n] = 1
                    A[i, i-1] = 1
                    A[i, i-1+n] = 1
                    A[i, i+n] = 1
                else:
                    A[i, i-1-n] = 1
                    A[i, i-n] = 1
                    A[i, i+1-n] = 1
                    A[i, i-1] = 1
                    A[i, i+1] = 1
                    A[i, i-1+n] = 1
                    A[i, i+n] = 1
                    A[i, i+1+n] = 1


    for i in range(n**2):
        A[i,i] = 1

    return A

predator1_adj = king_adj(predator1_view_range*2)
predator2_adj = king_adj(predator2_view_range*2)

predator1_adj = torch.tensor(predator1_adj).float()
predator2_adj = torch.tensor(predator2_adj).float()

class MADQN():  # def __init__(self,  dim_act, observation_state):
    def __init__(self, n_predator1, n_predator2, predator1_obs, predator2_obs, dim_act , buffer_size ,device = 'cpu'):
        self.shared_shape = shared_shape
        self.predator1_obs = predator1_obs
        self.predator2_obs = predator2_obs
        self.n_predator1 = n_predator1
        self.n_predator2 = n_predator2
        self.dim_act = dim_act
        self.epsilon = args.eps
        self.eps_decay = args.eps_decay
        self.device = device


        self.gdqns = [G_DQN_1(self.dim_act, self.predator1_obs).to(self.device) for _ in range(self.n_predator1)] + [
            G_DQN_2(self.dim_act, self.predator2_obs).to(self.device) for _ in range(self.n_predator2)]

        self.gdqn_targets = [G_DQN_1(self.dim_act, self.predator1_obs).to(self.device) for _ in range(self.n_predator1)] + [
            G_DQN_2(self.dim_act, self.predator2_obs).to(self.device) for _ in range(self.n_predator2)]

        self.buffers = [ReplayBuffer(capacity=buffer_size) for _ in range(self.n_predator1 + self.n_predator2)]
        self.gdqn_optimizers = [Adam(x.parameters(), lr=args.lr) for x in self.gdqns]

        self.criterion = nn.MSELoss()

        self.adj = None
        self.idx = None

        self.gdqn = None
        self.gdqn_target = None

        self.gdqn_optimizer = None
        self.target_optimizer = None

        self.buffer = None


    def target_update(self):
        weights = self.gdqn.state_dict()
        self.gdqn_target.load_state_dict(weights)


    def set_agent_info(self, agent):

        if agent[9] == "1":
            self.idx = int(agent[11:])
            self.adj = predator1_adj

        else:
            self.idx = int(agent[11:]) + n_predator1
            self.adj = predator2_adj

        self.gdqn= self.gdqns[self.idx]
        self.gdqn_target = self.gdqn_targets[self.idx]
        self.gdqn_optimizer = self.gdqn_optimizers[self.idx]
        self.buffer = self.buffers[self.idx]


    def set_agent_buffer(self ,idx):
        self.buffer = self.buffers[idx]

    def get_action(self, state, mask=None):

        try:
            torch.cuda.empty_cache()
        except:
            pass


        q_value = self.gdqn(torch.tensor(state).to(self.device), self.adj.to(self.device))

        #self.to_guestbook(shared_info.to('cpu'))
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)

        if np.random.random() < self.epsilon:
            print('random')
            # return random.randint(0, self.dim_act - 1), book
            return random.randint(0, dim_act-1)
        return torch.argmax(q_value).item()


    def replay(self):
        for _ in range(args.replay_times):

            self.gdqn_optimizer.zero_grad()

            observations,  actions, rewards, next_observations,  termination, truncation = self.buffer.sample()

            next_observations = torch.tensor(next_observations)
            observations = torch.tensor(observations)

            next_observations = next_observations.reshape(-1,self.shared_shape[2])
            observations = observations.reshape(-1,self.shared_shape[2])

            # to device
            observations = observations.to(self.device)
            next_observations = next_observations.to(self.device)
            adj = self.adj.to(self.device)

            q_values = self.gdqn(observations.unsqueeze(0), adj.unsqueeze(0))
            q_values = q_values[actions[0]]


            next_q_values= self.gdqn_target(next_observations.unsqueeze(0), adj.unsqueeze(0))
            next_q_values = torch.max(next_q_values)

            targets = int(rewards[0]) + (1 - int(termination[0])) * next_q_values * args.gamma
            loss = self.criterion(q_values, targets.detach())
            loss.backward()

            self.gdqn_optimizer.step()


            try:
                torch.cuda.empty_cache()
            except:
                pass

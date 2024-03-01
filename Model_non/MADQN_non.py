from model_non import G_DQN, ReplayBuffer
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

shared_shape = (args.map_size + predator1_view_range*2 ,args.map_size + predator1_view_range*2,3) #25+8+8=41
predator1_obs = (predator1_view_range*2,predator1_view_range*2,3)
predator2_obs = (predator2_view_range*2,predator2_view_range*2,3)

dim_act = 13

def king_adj(n) :

    A = np.zeros((n**2, n**2))

    for i in range(n**2):

        if i // n == 0  : #??? ?

            if i % n  == 0 : #??? ? : ?? 0 ??? ?

                A[i, i+1] = 1
                A[i, i+n] = 1
                A[i, i+1+n] = 1

            elif i % n == n-1 : #??? ?  ?? n-1 ??? ?

                A[i, i-1] = 1
                A[i, i-1+n] = 1
                A[i, i+n] = 1

            else:
                A[i, i-1] = 1
                A[i, i+1] = 1
                A[i, i-1+n] = 1
                A[i, i+n] = 1
                A[i, i+1+n] = 1

        elif i // n == n-1 : #??? ?

            if i % n ==  0 : #??? ?

                A[i, i-n] = 1
                A[i, i+1-n] = 1
                A[i, i+1] = 1
            elif i % n  == n-1 : #??? ?

                A[i, i-n] = 1
                A[i, i-1-n] = 1
                A[i, i-1] = 1
            else: #???
                A[i, i - 1] = 1
                A[i, i + 1] = 1
                A[i, i - 1-n] = 1
                A[i, i-n] = 1
                A[i, i+ 1-n] = 1

        else:  #?? ?? ???? ?



                if i % n == 0 : #??? ?
                    A[i, i-n] = 1
                    A[i, i+1-n] = 1
                    A[i, i+1] = 1
                    A[i, i+n] = 1
                    A[i, i+1+n] = 1

                elif i % n == n-1 : #??? ?
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



class MADQN():  # def __init__(self,  dim_act, observation_state):
    def __init__(self, n_predator1, n_predator2, predator1_obs, predator2_obs, dim_act , shared_shape, shared, device = 'cpu', buffer_size = 500):
        self.shared_shape = shared_shape
        self.predator1_obs = predator1_obs
        self.predator2_obs = predator2_obs
        self.n_predator1 = n_predator1
        self.n_predator2 = n_predator2
        self.dim_act = dim_act
        self.epsilon = args.eps
        self.eps_decay = args.eps_decay
        self.device = device

        # ?? n_predator1 ?? predator1? dqn ??, ? ?? ?? predator2 ? dqn ?? observation ? ??? ??? ?? dqn? ???? ??.
        self.gdqns = [G_DQN(self.dim_act, self.predator1_obs).to(self.device) for _ in range(self.n_predator1)] + [
            G_DQN(self.dim_act, self.predator2_obs).to(self.device) for _ in range(self.n_predator2)]
        self.gdqn_targets = [G_DQN(self.dim_act, self.predator1_obs).to(self.device) for _ in range(self.n_predator1)] + [
            G_DQN(self.dim_act, self.predator2_obs).to(self.device) for _ in range(self.n_predator2)]  # ??? ??? ?? target dqn ??
        self.buffers = [ReplayBuffer(capacity=buffer_size) for _ in range(self.n_predator1 + self.n_predator2)]
        self.gdqn_optimizers = [Adam(x.parameters(), lr=args.lr) for x in self.gdqns]
        #self.target_optimizer = [Adam(x.parameters(), lr=0.001) for x in self.gdqns_target] #?? ????? ?? ??? weight???? ???? ????
        self.criterion = nn.MSELoss()

        # shared gnn ? ??!
        self.shared = shared
        self.pos = None
        self.view_range = None


        self.idx = None

        self.gdqn = None
        self.gdqn_target = None

        self.gdqn_optimizer = None
        self.target_optimizer = None

        self.buffer = None
        self.observation_dim = None




    def target_update(self):
        weights = self.gdqn.state_dict()
        self.gdqn_target.load_state_dict(weights)


    def set_agent_info(self, agent, pos, view_range):

        if agent[9] == "1":
            self.idx = int(agent[11:])
            self.observation_dim = (view_range*2, view_range*2, 3)


            self.pos = pos
            self.view_range = view_range

        else:
            self.idx = int(agent[11:]) + n_predator1
            self.observation_dim = (view_range * 2, view_range * 2, 3)


            self.pos = pos
            self.view_range = view_range


        self.gdqn= self.gdqns[self.idx]
        self.gdqn_target = self.gdqn_targets[self.idx]
        self.gdqn_optimizer = self.gdqn_optimizers[self.idx]
        self.buffer = self.buffers[self.idx]


    def set_agent_buffer(self ,idx):
        self.buffer = self.buffers[idx]

    def set_agent_shared(self, view_range):
        self.view_range = view_range

    def from_guestbook(self):
        x_start = self.pos[1] + args.predator1_view_range -2
        y_start = self.pos[0] + args.predator1_view_range -2



        x_range = int(self.view_range)
        y_range = int(self.view_range)
        z_range = self.shared_shape[2]


        extracted_area = self.shared[x_start - (x_range -1):x_start + (x_range+1), y_start - (y_range -1): y_start + (y_range+1),
                         :z_range]

        return extracted_area


    def to_guestbook(self, info):

        x_start = self.pos[1] + args.predator1_view_range - 2
        y_start = self.pos[0] + args.predator1_view_range - 2

        x_range = int(self.view_range)
        y_range = int(self.view_range)


        self.shared[x_start - (x_range-1) :x_start + (x_range+1), y_start - (y_range - 1): y_start + (y_range+1), :] += info


    def shared_decay(self):
        self.shared = self.shared * args.book_decay


    def get_action(self, state, mask=None):

        # try:
        #     torch.cuda.empty_cache()
        # except:
        #     pass

        book = self.from_guestbook()
        q_value, shared_info = self.gdqn(torch.tensor(state).to(self.device), book.to(self.device))

        #self.to_guestbook(shared_info.to('cpu'))
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)

        if np.random.random() < self.epsilon:
            print('random')
            # return random.randint(0, self.dim_act - 1), book
            return random.randint(0, dim_act-1), book,shared_info
        return torch.argmax(q_value).item() , book , shared_info


        try:
            torch.cuda.empty_cache()
        except:
            pass




    def replay(self):
        for _ in range(args.replay_times):

            self.gdqn_optimizer.zero_grad()

            observations, book, actions, rewards, next_observations, book_next, termination, truncation = self.buffer.sample()

            next_observations = torch.tensor(next_observations)
            observations = torch.tensor(observations)

            next_observations = next_observations.reshape(-1,self.shared_shape[2])
            observations = observations.reshape(-1,self.shared_shape[2])

            # to device
            observations = observations.to(self.device)
            next_observations = next_observations.to(self.device)


            q_values, _ = self.gdqn(observations.reshape(self.observation_dim), book[0].detach().to(self.device))
            q_values = q_values[0][actions]


            next_q_values, _ = self.gdqn_target(next_observations.reshape(self.observation_dim), book_next[0].detach().to(self.device))
            next_q_values = torch.max(next_q_values)

            targets = int(rewards[0]) + (1 - int(termination[0])) * next_q_values * args.gamma
            loss = self.criterion(q_values, targets.detach())
            loss.backward()

            self.gdqn_optimizer.step()


            try:
                torch.cuda.empty_cache()
            except:
                pass
    def reset_shred(self,shared):
        self.shared = shared

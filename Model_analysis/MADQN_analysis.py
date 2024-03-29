from model_analysis import G_DQN, ReplayBuffer
import numpy as np
import random
import torch
import torch.nn as nn
from collections import deque
from torch.optim import Adam
from arguments import args
import wandb
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


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
    def __init__(self, n_predator1, n_predator2, predator1_obs, predator2_obs, dim_act , shared_shape, shared, buffer_size ,device = 'cpu'):
        self.shared_shape = shared_shape
        self.predator1_obs = predator1_obs
        self.predator2_obs = predator2_obs
        self.n_predator1 = n_predator1
        self.n_predator2 = n_predator2
        self.dim_act = dim_act
        self.epsilon = args.eps
        self.eps_decay = args.eps_decay
        self.device = device

        self.shared = shared
        self.pos = None
        self.view_range = None

        self.adj = None
        self.idx = None

        self.gdqn = None
        self.gdqn_target = None

        self.gdqn_optimizer = None
        self.target_optimizer = None

        self.buffer = None

        self.team_idx = None
        self.ep_move_count_pred = {
            0: 1,
            1: 1
        }

        self.step_move_count_pred = {
            0: 1,
            1: 1
        }

        self.summation_team_dist = {0: np.array([], dtype=float), 1: np.array([], dtype=float)}

        self.avg_dist_deque_pred1 = deque(maxlen=args.avg_dist_deque_len)
        self.avg_dist_deque_pred2 = deque(maxlen=args.avg_dist_deque_len)

        self.min_dist_deque_pred1 = deque(maxlen=args.avg_dist_deque_len)
        self.min_dist_deque_pred2 = deque(maxlen=args.avg_dist_deque_len)

        self.avg_move_deque_pred1 = deque(maxlen=args.avg_dist_deque_len)
        self.avg_move_deque_pred2 = deque(maxlen=args.avg_dist_deque_len)

        self.gdqns = [G_DQN(self.dim_act, self.predator1_obs).to(self.device) for _ in range(self.n_predator1)] + [
            G_DQN(self.dim_act, self.predator2_obs).to(self.device) for _ in range(self.n_predator2)]

        self.gdqn_targets = [G_DQN(self.dim_act, self.predator1_obs).to(self.device) for _ in range(self.n_predator1)] + [
            G_DQN(self.dim_act, self.predator2_obs).to(self.device) for _ in range(self.n_predator2)]

        self.buffers = [ReplayBuffer(capacity=buffer_size) for _ in range(self.n_predator1 + self.n_predator2)]
        self.gdqn_optimizers = [Adam(x.parameters(), lr=args.lr) for x in self.gdqns]

        self.criterion = nn.MSELoss()



    def target_update(self):
        weights = self.gdqn.state_dict()
        self.gdqn_target.load_state_dict(weights)

    def set_agent_model(self,agent):
        self.gdqn = self.gdqns[agent]
        self.gdqn_target = self.gdqn_targets[agent]




    def set_agent_info(self, agent, pos, view_range):

        if agent[9] == "1":
            self.idx = int(agent[11:])
            self.adj = predator1_adj

            self.pos = pos
            self.view_range = view_range


        else:
            self.idx = int(agent[11:]) + n_predator1
            self.adj = predator2_adj

            self.pos = pos
            self.view_range = view_range


        self.gdqn= self.gdqns[self.idx]
        self.gdqn_target = self.gdqn_targets[self.idx]
        self.gdqn_optimizer = self.gdqn_optimizers[self.idx]
        self.buffer = self.buffers[self.idx]


    def set_team_idx(self,idx):
        self.team_idx = idx


    # for ep team move counting
    def ep_move_count(self):
        self.ep_move_count_pred[self.team_idx] += 1

    def reset_ep_move_count(self):
        self.ep_move_count_pred[0] = 1
        self.ep_move_count_pred[1] = 1

    # for step team move counting
    # 매 스텝이 시작할때마다 reset 을 해주어야함
    def step_move_count(self):
        self.step_move_count_pred[self.team_idx] += 1

    def reset_step_move_count(self):
        self.step_move_count_pred[0] = 1
        self.step_move_count_pred[1] = 1


    def set_agent_buffer(self ,idx):
        self.buffer = self.buffers[idx]

    def set_agent_pos(self, pos):
        self.pos = pos

    def set_agent_shared(self, view_range):
        self.view_range = view_range


    # must call this function after calling 'set_agent_info'


    def dist(self, obs):
        # prey position
        prey_pos = np.argwhere(obs[:,:,2] == 1)

        # predator position
        pred_pos = np.array([[(self.view_range*2 - 1)/2, (self.view_range*2 - 1)/2]], dtype=np.int32)

        # entire position
        entire_pos = np.append(pred_pos, prey_pos ,axis=0)

        # calcualate distance
        distance_matrix = squareform(pdist(entire_pos, 'euclidean'))

        return distance_matrix[0,1:]

    def put_dist(self, dist_list):

        self.summation_team_dist[self.team_idx] = np.concatenate((self.summation_team_dist[self.team_idx],dist_list))

    def reset_summation_team_dist(self):
        self.summation_team_dist[0] = np.array([], dtype=float)
        self.summation_team_dist[1] = np.array([], dtype=float)



    def avg_dist(self, obs):
        # prey position
        prey_pos = np.argwhere(obs[:,:,2] == 1)

        # predator position
        pred_pos = np.array([[(self.view_range*2 - 1)/2, (self.view_range*2 - 1)/2]], dtype=np.int32)

        # entire position
        entire_pos = np.append(pred_pos, prey_pos ,axis=0)

        # calcualate distance
        distance_matrix = squareform(pdist(entire_pos, 'euclidean'))

        return np.mean(distance_matrix[0,1:])

    def min_dist(self, obs):
        # prey position
        prey_pos = np.argwhere(obs[:, :, 2] == 1)

        # predator position
        pred_pos = np.array([[(self.view_range * 2 - 1) / 2, (self.view_range * 2 - 1) / 2]], dtype=np.int32)

        # entire position
        entire_pos = np.append(pred_pos, prey_pos, axis=0)

        # calcualate distance
        distance_matrix = squareform(pdist(entire_pos, 'euclidean'))

        return np.min(distance_matrix[0, 1:])

    def from_guestbook(self):
        x_start = self.pos[1] + args.predator1_view_range -2
        y_start = self.pos[0] + args.predator1_view_range -2



        x_range = int(self.view_range)
        y_range = int(self.view_range)
        z_range = self.shared_shape[2]


        extracted_area = self.shared[x_start - (x_range -1):x_start + (x_range+1), y_start - (y_range -1): y_start + (y_range+1),
                         :z_range]

        return extracted_area


    # def to_guestbook(self, info):
    #
    #     x_start = self.pos[1] + args.predator1_view_range - 2
    #     y_start = self.pos[0] + args.predator1_view_range - 2
    #
    #     x_range = int(self.view_range)
    #     y_range = int(self.view_range)
    #
    #
    #     self.shared[x_start - (x_range-1) :x_start + (x_range+1), y_start - (y_range - 1): y_start + (y_range+1), :] += info
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
        q_value, shared_info = self.gdqn(torch.tensor(state).to(self.device), self.adj.to(self.device), book.to(self.device))

        #self.to_guestbook(shared_info.to('cpu'))
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)

        if np.random.random() < self.epsilon:
            print('random')
            # return random.randint(0, self.dim_act - 1), book
            return random.randint(0, dim_act-1), book,shared_info
        return torch.argmax(q_value).item(), book, shared_info

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
            adj = self.adj.to(self.device)

            q_values, _ = self.gdqn(observations.unsqueeze(0), adj.unsqueeze(0), book[0].detach().to(self.device))
            q_values = q_values[actions]


            next_q_values, _ = self.gdqn_target(next_observations.unsqueeze(0), adj.unsqueeze(0), book_next[0].detach().to(self.device))
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



    def avg_dist_into_deque_pred1(self, dist):
        self.avg_dist_deque_pred1.append(dist)

    def avg_dist_into_deque_pred2(self, dist):
        self.avg_dist_deque_pred2.append(dist)


    def min_dist_into_deque_pred1(self, dist):
        self.min_dist_deque_pred1.append(dist)


    def min_dist_into_deque_pred2(self, dist):
        self.min_dist_deque_pred2.append(dist)


    def avg_move_into_deque_pred1(self, move):
        self.avg_move_deque_pred1.append(move)

    def avg_move_into_deque_pred2(self, move):
        self.avg_move_deque_pred2.append(move)


    def plot(self,ep):
        avg_dist_list_pred1 = list(self.avg_dist_deque_pred1)
        avg_dist_list_pred2 = list(self.avg_dist_deque_pred2)

        avg_move_list_pred1 = list(self.avg_move_deque_pred1)
        avg_move_list_pred2 = list(self.avg_move_deque_pred2)

        min_dist_list_pred1 = list(self.min_dist_deque_pred1)
        min_dist_list_pred2 = list(self.min_dist_deque_pred2)


        # avg(distance) - avg(move)
        plt.figure(figsize=(10, 6))
        plt.scatter(avg_dist_list_pred1, avg_move_list_pred1, facecolors='none', edgecolors='blue', label='pred1')
        plt.scatter(avg_dist_list_pred2, avg_move_list_pred2, facecolors='none', edgecolors='red', label='pred2')

        plt.title('avg move for the avg distance')
        plt.xlabel('avg_dist')
        plt.ylabel('avg_move')
        plt.legend()


        wandb.log({"avg move for the avg distance_ep_{}".format(ep): wandb.Image(plt)})

        plt.close()  # 현재 그림 닫기

        # min(distance) - avg(move)
        plt.figure(figsize=(10, 6))
        plt.scatter(min_dist_list_pred1, avg_move_list_pred1, facecolors='none', edgecolors='blue', label='pred1')
        plt.scatter(min_dist_list_pred2, avg_move_list_pred2, facecolors='none', edgecolors='red', label='pred2')

        plt.title('min move for the avg distance')
        plt.xlabel('min_dist')
        plt.ylabel('avg_move')
        plt.legend()


        wandb.log({"min move for the avg distance_ep_{}".format(ep): wandb.Image(plt)})

        plt.close()  # 현재 그림 닫기



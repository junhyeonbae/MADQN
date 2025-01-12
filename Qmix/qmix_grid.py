import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mix_net_grid import QMIX_Net, VDN_Net


# orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)



class Q_network_MLP(nn.Module):
    def __init__(self, args, input_dim): #input demension 이 어떻게 되는건지?
        super(Q_network_MLP, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, inputs):
        # When 'choose_action', inputs.shape(N,input_dim)
        # When 'train', inputs.shape(bach_size,max_episode_len,N,input_dim)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        Q = self.fc3(x)
        return Q


class QMIX_SMAC(object):
    def __init__(self, args):
        self.N = args.N
        self.N = args.n_predator1 + args.n_predator2  #check!
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.add_last_action = args.add_last_action
        self.add_agent_id = args.add_agent_id
        self.max_train_steps=args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.use_grad_clip = args.use_grad_clip
        self.batch_size = args.batch_size  # 这里的batch_size代表有多少个episode
        self.target_update_freq = args.target_update_freq
        self.tau = args.tau
        self.use_hard_update = args.use_hard_update
        #self.use_rnn = args.use_rnn
        self.algorithm = args.algorithm
        self.use_double_q = args.use_double_q
        self.use_RMS = args.use_RMS
        self.use_lr_decay = args.use_lr_decay
        self.episode_len = args.episode_len

        # Compute the input dimension -> 주 모델의 input이 last_action 과 agent id 를 포함하지 않기 때문에 포함시키지 않음
        self.input_dim = self.obs_dim
        # if self.add_last_action:
        #     print("------add last action------")
        #     self.input_dim += self.action_dim
        #
        #
        # if self.add_agent_id:
        #     print("------add agent id------")
        #     self.input_dim += self.N


        print("------use MLP------")
        self.eval_Q_net = Q_network_MLP(args, self.input_dim)
        self.target_Q_net = Q_network_MLP(args, self.input_dim)
        self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())



        if self.algorithm == "QMIX":
            print("------algorithm: QMIX------")
            self.eval_mix_net = QMIX_Net(args)
            self.target_mix_net = QMIX_Net(args)
        elif self.algorithm == "VDN":
            print("------algorithm: VDN------")
            self.eval_mix_net = VDN_Net()
            self.target_mix_net = VDN_Net()
        else:
            print("wrong!!!")
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())



        self.eval_parameters = list(self.eval_mix_net.parameters()) + list(self.eval_Q_net.parameters())
        if self.use_RMS:
            print("------optimizer: RMSprop------")
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.lr)
        else:
            print("------optimizer: Adam------")
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=self.lr)

        self.train_step = 0


    # choose_action : step 단위마다 call 되는 함수
    def choose_action(self, obs_n, last_onehot_a_n, avail_a_n, epsilon):  # obs_n.shape=(N,obs_dim) last_onehot_a_n.shape=(N,action_dim) avail_a_n.shape=(N,action_dim)
        with torch.no_grad(): #단지 현재 obs_n 에서 action 을 뽑는것이 목적이므로 메모리를 사용할 필요가 없음
            if np.random.uniform() < epsilon:  # epsilon-greedy
                # Only available actions can be chosen
                a_n = [np.random.randint(low=0, high=self.action_dim, dtype=int) for _ in range(self.N)] #[ ] 리스트 형태로 에이전트들 임의의 행동 뽑기
            else:
                inputs = []
                obs_n = torch.tensor(obs_n, dtype=torch.float32)  # obs_n.shape=(N，obs_dim) obs_n이 그냥 하나의 벡터로 나오는 것 같다. 그렇다면 gridworld에서는 flatten 시켜서 사용해야 하나?
                                                                  # 텐서 변환 시켜주고
                inputs.append(obs_n)
                # if self.add_last_action:
                #     last_a_n = torch.tensor(last_onehot_a_n, dtype=torch.float32)
                #     inputs.append(last_a_n)
                # if self.add_agent_id:
                #     inputs.append(torch.eye(self.N))

                #inputs = torch.cat([x for x in inputs], dim=-1)  # inputs.shape=(N,inputs_dim) 위에서 add_last_action,agent_id 넣지 않는 이상 이건 필요없을것 같음
                q_value = self.eval_Q_net(inputs)
                  # avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)  # avail_a_n.shape=(N, action_dim)
                # q_value[avail_a_n == 0] = -float('inf')  # Mask the unavailable actions
                a_n = q_value.argmax(dim=-1).numpy()
            return a_n

    def train(self, replay_buffer, total_steps):
        batch, max_episode_len = replay_buffer.sample()  # Get training data
        self.train_step += 1


        #batch 구성 : obs_n, s 는 (len_episode +1)의 길이로, 나머지는 (len_episode) 의 길이로 dictioinary에 저장되어 있다.
        # get_inputs 함수 내부적으로는 사실상 obs_n 만 가져오는 것이다 .
        inputs = self.get_inputs(batch, max_episode_len)  # inputs.shape=(bach_size,max_episode_len+1,N,input_dim)

        q_evals = self.eval_Q_net(inputs[:, :-1])  # q_evals.shape=(batch_size,max_episode_len,N,action_dim) 모든 에피소드의 정보를 가져오는데, 근데 이제 마지막 스텝은 제외한 것
        q_targets = self.target_Q_net(inputs[:, 1:]) # 모든 에피소드의 정보를 가져오는데, 근데 이제 초기 step은 제외한 것

        with torch.no_grad(): #그래디언트 필요없는 연산해서 메모리 아끼기 위함
            if self.use_double_q:  # If use double q-learning, we use eval_net to choose actions,and use target_net to compute q_target
                q_eval_last = self.eval_Q_net(inputs[:, -1].reshape(-1, self.input_dim)).reshape(self.batch_size, 1, self.N, -1) #마지막 스텝에 대해서만 하는건데,,,
                q_evals_next = torch.cat([q_evals[:, 1:], q_eval_last], dim=1) # q_evals_next.shape=(batch_size,max_episode_len,N,action_dim)
                q_evals_next[batch['avail_a_n'][:, 1:] == 0] = -999999
                a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)  # a_max.shape=(batch_size,max_episode_len, N, 1)
                q_targets = torch.gather(q_targets, dim=-1, index=a_argmax).squeeze(-1)  # q_targets.shape=(batch_size, max_episode_len, N)
            else:
                # q_targets[batch['avail_a_n'][:, 1:] == 0] = -999999 #가능하지 않은 action의 q값에는 매우 작은 q값을 배치
                q_targets = q_targets.max(dim=-1)[0]  # q_targets.shape=(batch_size, max_episode_len, N) #q_targets의 마지막 차원의 관점에서 가장 큰 값을 찾으라는 말이다.

        # batch['a_n'].shape(batch_size,max_episode_len, N)
        q_evals = torch.gather(q_evals, dim=-1, index=batch['a_n'].unsqueeze(-1)).squeeze(-1)  # q_evals.shape(batch_size, max_episode_len, N) # (y-Q value)^2 의 값에서 q value 에 실제 an 값 넣어서 Q value 구하는 것

        # Compute q_total using QMIX or VDN, q_total.shape=(batch_size, max_episode_len, 1)
        if self.algorithm == "QMIX":
            q_total_eval = self.eval_mix_net(q_evals, batch['s'][:, :-1])
            q_total_target = self.target_mix_net(q_targets, batch['s'][:, 1:])
        else:
            q_total_eval = self.eval_mix_net(q_evals)
            q_total_target = self.target_mix_net(q_targets)
        # targets.shape=(batch_size,max_episode_len,1)
        targets = batch['r'] + self.gamma * (1 - batch['dw']) * q_total_target

        td_error = (q_total_eval - targets.detach())
        # mask_td_error = td_error * batch['active']
        # loss = (mask_td_error ** 2).sum() / batch['active'].sum() #유효한 time step 의 수로 나누어 평균을 구한 것
        loss = (td_error ** 2).sum() / (self.batch_size * self.episode_len) # 유효한 time step 의 수로 나누어 평균을 구한 것
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.eval_parameters, 10)
        self.optimizer.step()

        if self.use_hard_update: # 내 모델도 hard update 이므로 이걸로 하면 됨
            # hard update
            if self.train_step % self.target_update_freq == 0:
                self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())
                self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        else:
            # Softly update the target networks
            for param, target_param in zip(self.eval_Q_net.parameters(), self.target_Q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.eval_mix_net.parameters(), self.target_mix_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    #     if self.use_lr_decay: # 내 모델에서 이거 안하니까 안해도 될 듯?
    #         self.lr_decay(total_steps)
    #
    # def lr_decay(self, total_steps):  # Learning rate Decay
    #     lr_now = self.lr * (1 - total_steps / self.max_train_steps)
    #     for p in self.optimizer.param_groups:
    #         p['lr'] = lr_now

    def get_inputs(self, batch, max_episode_len):
        inputs = []
        inputs.append(batch['obs_n'])
        # if self.add_last_action: #이건 없어도 될 것 같고
        #     inputs.append(batch['last_onehot_a_n'])
        # if self.add_agent_id: # 에이전트의 위치값은 필요하니까 이는 필요할 것 같다.
        #     agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, max_episode_len + 1, 1, 1)
        #     inputs.append(agent_id_one_hot)

        # inputs.shape=(bach_size,max_episode_len+1,N,input_dim)  # +1 은 초깃값 때문에 있는거
        # inputs = torch.cat([x for x in inputs], dim=-1)

        return inputs

    def save_model(self, env_name, algorithm, number, seed, total_steps):
        torch.save(self.eval_Q_net.state_dict(), "./model/{}/{}_eval_rnn_number_{}_seed_{}_step_{}k.pth".format(env_name, algorithm, number, seed, int(total_steps / 1000)))

import numpy as np
import torch
import copy


class ReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.episode_limit = args.episode_limit # Maximum number of steps per episode
        self.buffer_size = args.buffer_size #buffer 사이즈를 미리 결정해놓음-->사실상 episode의 개
        self.batch_size = args.batch_size # Batch size (the number of episodes)
        self.episode_num = 0
        self.current_size = 0
        # 에이전트 수만큼 받아야 하는 것들은 4차원이고, 그 이외의 것들은 3차원:  사실상 buffer_size == 에피소드 개수
        self.buffer = {'obs_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.N, self.obs_dim]),
                       's': np.zeros([self.buffer_size, self.episode_limit + 1, self.state_dim]),
                       'avail_a_n': np.ones([self.buffer_size, self.episode_limit + 1, self.N, self.action_dim]),  # Note: We use 'np.ones' to initialize 'avail_a_n'
                       'last_onehot_a_n': np.zeros([self.buffer_size, self.episode_limit + 1, self.N, self.action_dim]),
                       'a_n': np.zeros([self.buffer_size, self.episode_limit, self.N]),
                       'r': np.zeros([self.buffer_size, self.episode_limit, 1]), # 전체 리워드를 받기 때문에 step 당 r는 하나만 필요한 것!
                       'dw': np.ones([self.buffer_size, self.episode_limit, 1]),  # Note: We use 'np.ones' to initialize 'dw' --> 에피소드가 완료되었는지의 여부를 나타내는 플래그, 에피소드가 끝났다면, 해당 타임스텝에서 dw =1 으로 설정됨
                       'active': np.zeros([self.buffer_size, self.episode_limit, 1]) # 에드소드별로 길이가 다를 수 있기 때문에, 에피소드가 끝난 후에도 남은 타임스텝을 masking 하는 목적이다.
                       }
        self.episode_len = np.zeros(self.buffer_size) #각 에피소드의 길이를 입력하기 위함

    def store_transition(self, episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw): #버퍼 만들때는 buffer_size로 만들었는데, 이는 사실 episode 이고, 그래서 저장할때는 episode_step 으로 넣어주는 것!
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.buffer['last_onehot_a_n'][self.episode_num][episode_step + 1] = last_onehot_a_n
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['r'][self.episode_num][episode_step] = r
        self.buffer['dw'][self.episode_num][episode_step] = dw

        self.buffer['active'][self.episode_num][episode_step] = 1.0  #transition 을 저장한다는 말 자체가 step 이 살아있다는 것이므로 1을 저장한다.

    def store_last_step(self, episode_step, obs_n, s, avail_a_n): # 각 에피소드의 마지막에는 r, dw, active 가 필요없음
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['avail_a_n'][self.episode_num][episode_step] = avail_a_n
        self.episode_len[self.episode_num] = episode_step  # Record the length of this episode
        self.episode_num = (self.episode_num + 1) % self.buffer_size #이게 꼭 돌아가야지 episode_num이 갱신된다!! 나머지? 버퍼가 최대사이즈가 되면 다시 0으로 돌아가서 채우게 된다.
        self.current_size = min(self.current_size + 1, self.buffer_size) # 위에서 설정해놓은 버퍼 사이즈에 꽉 채워지지 않을 수도 있어서 둘 중 작은 값을 버퍼의 현재 사이즈라고 하는 것

    def sample(self):
        # Randomly sampling
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False) #아, buffer_size가 아니라 batch_size구나..
        max_episode_len = int(np.max(self.episode_len[index])) #각 에피소드드의 최장 길이를 저장해놓고... 사실 grid world 에서는 정해져있으니까 굳이 필요 없긴 함!
        batch = {} # 딕셔너리
        for key in self.buffer.keys():
            if key == 'obs_n' or key == 's' or key == 'avail_a_n' or key == 'last_onehot_a_n':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len + 1], dtype=torch.float32) #여기서 +1 이 있는이유는 obs_n, s 등등은 마지막 step 에서의 것도 저장했기 때문!
            elif key == 'a_n':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.float32)

        return batch, max_episode_len

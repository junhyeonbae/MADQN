import torch
import numpy as np
import argparse
from replay_buffer_grid import ReplayBuffer
from qmix_grid import QMIX_SMAC

from magent2.environments import hetero_adversarial_v1
from arguments import args

#render_mode = 'human'
render_mode = 'rgb_array'
env = hetero_adversarial_v1.env(map_size=args.map_size, minimap_mode=False, tag_penalty=args.tag_penalty,
                                    max_cycles=args.max_update_steps, extra_features=False, render_mode=render_mode)



from torch.utils.tensorboard import SummaryWriter
from smac.env import StarCraft2Env
from normalization import Normalization



env = hetero_adversarial_v1.env(map_size=args.map_size, minimap_mode=False, tag_penalty=args.tag_penalty,
                                    max_cycles=args.max_update_steps, extra_features=False, render_mode=render_mode)


class Runner_QMIX_SMAC:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        # self.env = StarCraft2Env(map_name=self.env_name, seed=self.seed)
        self.env = hetero_adversarial_v1.env(map_size=args.map_size, minimap_mode=False, tag_penalty=args.tag_penalty,
                                    max_cycles=args.max_update_steps, extra_features=False, render_mode=render_mode)
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]  # The number of agents
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        self.agent_n = QMIX_SMAC(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='./runs/{}/{}_env_{}_number_{}_seed_{}'.format(self.args.algorithm, self.args.algorithm, self.env_name, self.number, self.seed))

        self.epsilon = self.args.epsilon  # Initialize the epsilon
        self.win_rates = []  # Record the win rates
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)

    def run(self, ): #이게 가장 중요한 함수이다.
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:  # 여기서 총 episode 개수가 결정되는것이다 --> max_train_steps에 따라 몇 번의 episode가 돌아가는지 결정되는것
            # if self.total_steps // self.args.evaluate_freq > evaluate_num: # 정책평가 기준 스텝때마다 정책평가 --> smac이 아닌이상 사실상 필요없는 코드
            #     self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps 마다 policy 성능 측정하고...
            #     evaluate_num += 1

            _, _, episode_steps = self.run_episode_smac(evaluate=False)  # Run an episode #여기서 뱉는 episode_steps 이 뭔지 확인해야 함
            self.total_steps += episode_steps #episode가 끝나면 전체가 몇 step 인지 total_step 에 더해준다.

            if self.replay_buffer.current_size >= self.args.batch_size: #배치 사이즈보다 buffer가 크면 학습을 시작한다. 그러니까 episode 끝나면 한번씩 업데이트하는 것임
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training

        # self.evaluate_policy() #마지막으로 정책 평가하고
        self.env.close() #환경 닫음

    # def evaluate_policy(self, ): #smac 에서 policy 얼마나 좋은지 그냥 측정하는거고.. 그래서 없어도 된다.
    #     win_times = 0
    #     evaluate_reward = 0
    #     for _ in range(self.args.evaluate_times):
    #         win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True)
    #         if win_tag:
    #             win_times += 1
    #         evaluate_reward += episode_reward
    #
    #     win_rate = win_times / self.args.evaluate_times
    #     evaluate_reward = evaluate_reward / self.args.evaluate_times
    #     self.win_rates.append(win_rate)
    #     print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(self.total_steps, win_rate, evaluate_reward))
    #     self.writer.add_scalar('win_rate_{}'.format(self.env_name), win_rate, global_step=self.total_steps)
    #     # Save the win rates
    #     np.save('./data_train/{}_env_{}_number_{}_seed_{}.npy'.format(self.args.algorithm, self.env_name, self.number, self.seed), np.array(self.win_rates))

    def run_episode_smac(self, evaluate=False): #에피소드 하나 돌리는 함수

        episode_reward = 0
        self.env.reset() # 여기서 환경 reset 을 해주고!
        # if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
        #     self.agent_n.eval_Q_net.rnn_hidden = None
        # last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))  # Last actions of N agents(one-hot) # 이것도 grid, 내 환경에서는 필요없는 세팅


        # for episode_step in range(self.args.episode_limit): # episode_limit : 하나의 에피소드에서 최대한 돌아가는 steps 이다. 이건 처음에 조건 형성할 때, episode * total_step으로 해야 한다.
        for agent in env.agent_iter():

            iteration_number = 0

            if agent[:8] == "predator":

                if agent[9] == "1":
                    agent_idx = int(agent[11:])
                else:
                    agent_idx = int(agent[11:])


            observation, reward, termination, truncation, info = env.last()  # 가장 최근의 상태를 업데이트 해주는 것
            flatten_tensor = observation.view(-1)

            # obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
            # s = self.env.get_state()  # s.shape=(state_dim,)
            # avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim) #smac에서는 특정 상황에서 불가능한 행동이 있을 수도 있는 것 같음
            epsilon = 0 if evaluate else self.epsilon #평가할때는 epsilon = 0 , 학습할때는 epsilon 값 살아있음!

            a_n = self.agent_n.choose_action(observation, epsilon)
            env.step(a_n)
            # last_onehot_a_n = np.eye(self.args.action_dim)[a_n]  # Convert actions to one-hot vectors
            # r, done, info = self.env.step(a_n)  # Take a step
            # win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += reward

            if not evaluate: #평가하는게 아니라면, 매번 replay buffer에 experience를 저장
                if self.args.use_reward_norm: #reward 를 norm 한다는데 필요없을 것 같음
                    r = self.reward_norm(r)
                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if truncation : #끝났고
                    dw = True
                else:
                    dw = False

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw) #episode_step 이 episode 번호인가?
                # Decay the epsilon
                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min

            if done:
                break

            iteration_number += 1

        if not evaluate: #특정 episode가 끝난뒤에 마지막 obs, s, avail_a_n 과 해당 episode에 몇 번의 step을 돌았는지 뱉는것
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            self.replay_buffer.store_last_step(episode_step + 1, obs_n, s, avail_a_n)

        return win_tag, episode_reward, episode_step + 1

#while self.total_steps < self.args.max_train_steps:
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for QMIX and VDN in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=32, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")
    parser.add_argument("--episode_len", type=int, default=int(1e5), help="steps per one episode")

    parser.add_argument("--algorithm", type=str, default="QMIX", help="QMIX or VDN")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_decay_steps", type=float, default=50000, help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--buffer_size", type=int, default=5000, help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--qmix_hidden_dim", type=int, default=32, help="The dimension of the hidden layer of the QMIX network")
    parser.add_argument("--hyper_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the hyper-network")
    parser.add_argument("--hyper_layers_num", type=int, default=1, help="The number of layers of hyper-network")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="use lr decay")
    parser.add_argument("--use_RMS", type=bool, default=False, help="Whether to use RMS,if False, we will use Adam")
    parser.add_argument("--add_last_action", type=bool, default=True, help="Whether to add last actions into the observation")
    parser.add_argument("--add_agent_id", type=bool, default=True, help="Whether to add agent id into the observation")
    parser.add_argument("--use_double_q", type=bool, default=False, help="Whether to use double q-learning")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Whether to use reward normalization")
    parser.add_argument("--use_hard_update", type=bool, default=True, help="Whether to use hard update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network")
    parser.add_argument("--tau", type=int, default=0.005, help="If use soft update")

    parser.add_argument('--dim_feature', type=int, default=4)

    parser.add_argument('--total_ep', type=int, default=10000)
    parser.add_argument('--ep_save', type=int, default=1000)
    parser.add_argument('--jitter_std', type=float, default=0.5)

    # env = hetero_adversarial_v1.env(map_size=args.map_size, minimap_mode=False, tag_penalty=args.tag_penalty,
    #                                 max_cycles=args.max_update_steps, extra_features=False, render_mode=render_mode)

    parser.add_argument('--max_update_steps', type=int, default=4) # 한 episode 당 돌아가는 에피소드 개수... 사실상 에이전트 종료하는 term까지 포함해서 +1 돌아간다.
    parser.add_argument('--trainstart_buffersize', type=int, default=5000)
    parser.add_argument('--replay_times', type=float, default=32)
    parser.add_argument('--target_update', type=float, default=10)

    parser.add_argument('--map_size', type=int, default=24)
    parser.add_argument('--predator1_view_range', type=int, default=10)
    parser.add_argument('--predator2_view_range', type=int, default=5)
    parser.add_argument('--n_predator1', type=int, default=2)
    parser.add_argument('--n_predator2', type=int, default=2)
    parser.add_argument('--n_prey', type=int, default=3)
    parser.add_argument('--tag_reward', type=float, default=3)
    parser.add_argument('--tag_penalty', type=float, default=-0.2)
    parser.add_argument('--move_penalty', type=float, default=-0.15)

    parser.add_argument('--seed', type=int, default=874)

    args = parser.parse_args()
    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps

    env_names = ['3m', '8m', '2s3z']
    env_index = 0
    runner = Runner_QMIX_SMAC(args, env_name=env_names[env_index], number=1, seed=0)
    runner.run()


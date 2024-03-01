import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--eps', type=float, default=1)
parser.add_argument('--eps_decay', type=float, default=0.9999)
parser.add_argument('--eps_min', type=float, default=0.01)
parser.add_argument('--max_update_steps', type=int, default=100)
parser.add_argument('--total_ep', type=int, default=10000)
parser.add_argument('--book_decay',type=int, default=0.5)
parser.add_argument('--ep_save',type=int, default = 100)


parser.add_argument('--buffer_size', type=int, default=10000)
parser.add_argument('--trainstart_buffersize', type=int, default=5000)

parser.add_argument('--replay_times',type=float, default= 32)
parser.add_argument('--target_update',type=float, default= 1)

parser.add_argument('--map_size',type=int, default=14)
parser.add_argument('--predator1_view_range',type=int, default=6)
parser.add_argument('--predator2_view_range',type=int, default=3)
parser.add_argument('--n_predator1',type=int, default=3)
parser.add_argument('--n_predator2',type=int, default=3)
parser.add_argument('--n_prey',type=int, default=12)
parser.add_argument('--tag_reward',type=float, default= 3)
parser.add_argument('--tag_penalty',type=float, default= -1)

parser.add_argument('--seed',type=int, default=874)


args = parser.parse_args()
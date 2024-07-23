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
from random import randint



predator1_view_range = args.predator1_view_range
predator2_view_range = args.predator2_view_range
n_predator1 = args.n_predator1
n_predator2 = args.n_predator2
n_prey = args.n_prey
dim_feature = args.dim_feature


shared_shape = (args.map_size + (predator1_view_range-2)*2, args.map_size + (predator1_view_range-2)*2, 4)
predator1_obs = (predator1_view_range*2,predator1_view_range*2,4)
predator2_obs = (predator2_view_range*2,predator2_view_range*2,4)

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


def overlap_area(a, b, x, y, side_length1, side_length2):
    overlap_width = min(a + side_length1 / 2, x + side_length2 / 2) - max(a - side_length1 / 2, x - side_length2 / 2)
    overlap_height = min(b + side_length1 / 2, y + side_length2 / 2) - max(b - side_length1 / 2, y - side_length2 / 2)

    if overlap_width > 0 and overlap_height > 0:
        return overlap_width * overlap_height
    else:
        return 0


#한번의 step 에서는 pos가 변하지 않기 때문에 사실 한번만 실행해도 되는 함수이긴한데,,,
#매 step 이 아니라, last step 을 제외한 부분에서만 실행하면 될 것 같다.
# result matrix 의
def calculate_Overlap_ratio(pos):

    centers = pos
    centers = [(x + 1, y + 1) for x, y in centers]

    side_lengths = [predator1_view_range*2 for i in range(n_predator1)] + [predator2_view_range*2 for i in range(n_predator2)]

    overlap_matrix = np.zeros((n_predator1+n_predator2, n_predator1+n_predator2))
    square_areas = np.zeros(n_predator1+n_predator2)  # 각 정사각형의 넓이를 저장할 배열

    # 각 정사각형의 넓이 계산
    for i in range(n_predator1+n_predator2):
        square_areas[i] = side_lengths[i] ** 2

    # 겹치는 영역의 넓이 계산
    for i in range(n_predator1+n_predator2): #행의 기준이 되는 에이전트를 의미함
        for j in range(i + 1, n_predator1+n_predator2):
            a, b = centers[i]
            x, y = centers[j]
            overlap = overlap_area(a, b, x, y, side_lengths[i], side_lengths[j])
            overlap_matrix[i, j] = overlap
            overlap_matrix[j, i] = overlap

    # 각 행을 해당 정사각형의 넓이로 나눔
    for i in range(n_predator1+n_predator2):
        overlap_matrix[i, :] /= square_areas[i]


    # 결과 행렬 초기화
    result_matrix = np.zeros((n_predator1+n_predator2, 2))

    result_matrix[:, 0] = np.sum(overlap_matrix[:, :n_predator1], axis=1)  # 첫 3개 정사각형에 대한 합
    result_matrix[:, 1] = np.sum(overlap_matrix[:, n_predator2:], axis=1)  # 나머지 3개 정사각형에 대한 합

    return result_matrix

def calculate_Overlap_ratio_intake(past, now):

    centers_past = past
    centers_now = now

    centers_past = [(x + 1, y + 1) for x, y in centers_past]
    centers_now = [(x + 1, y + 1) for x, y in centers_now]

    side_lengths = [predator1_view_range*2 for i in range(n_predator1)] + [predator2_view_range*2 for i in range(n_predator2)]

    overlap_matrix = np.zeros((n_predator1+n_predator2, n_predator1+n_predator2))


    # 겹치는 영역의 넓이 계산
    for i in range(n_predator1+n_predator2): #행의 기준이 되는 에이전트를 의미함
        for j in range(n_predator1+n_predator2):
            a, b = centers_now[i]
            x, y = centers_past[j]
            overlap = overlap_area(a, b, x, y, side_lengths[i], side_lengths[j])
            overlap_matrix[i, j] = overlap


    # 결과 행렬 초기화
    result_matrix = np.zeros((n_predator1+n_predator2, 2))

    result_matrix[:, 0] = np.sum(overlap_matrix[:, :n_predator1], axis=1)  # 첫 3개 정사각형에 대한 합
    result_matrix[:, 1] = np.sum(overlap_matrix[:, n_predator2:], axis=1)  # 나머지 3개 정사각형에 대한 합

    return result_matrix



def prey_number(observation_temp):
    prey_pos = np.argwhere(observation_temp[:, :, args.dim_feature - 1] == 1)

    return len(prey_pos)



# prey 에 있는 number 의 set 을 만든다.
def prey_num_set(prey_deque):

    num_set = set()
    for deque_values in prey_deque.values():
        num_set.update(deque_values)

    return num_set

# prey number set 을 이용하여 각각의 숫자가 어느곳에 위치하는지 찾는다.
# {0:[(1,1),(3,5),(1,5)],1:[(1,3),(2,5),(1,4)]} 등과 같이 저장이 되는것이다.
def find_prey_positions(prey_num_set, prey_deque):

    num_set = prey_num_set
    positions = {number: [] for number in num_set}

    # 각각의 prey 수가 담겨있는 위치가 어디인지 딕셔너리에 넣는 작업
    for agent_idx, deque_values in prey_deque.items():
        for idx, prey in enumerate(deque_values):
            positions[prey].append((agent_idx, idx))

    return positions

# find_numbers_positions 함수로부터 얻는 위치값을 l2_deque 에 적용하여 해당 값을 알아낸다.
def extract_values_based_on_positions(prey_set,positions_dict, l2_before, l2_deque):
    pos_dict = positions_dict
    results = {number: [] for number in prey_set}  # 결과를 숫자별로 저장할 딕셔너리

    # pos_dict을 이용하여 결과 수집
    for number, pos_list in pos_dict.items(): #pos_dict 에 있는 키와 쌍을 반환, 즉 number에는 숫자가 pop_list 에는 각 숫자가 있는 position 이 찍힌다.
        for pos in pos_list:
            x, y = pos #agent_idx 에는 position의 x축 값이, idx 에는 y 값이 찍힌다.
            corresponding_l2_before = l2_before[x][y]
            corresponding_l2 = l2_deque[x][y] #corresponding_value 에는 해당값이 찍힌다.
            results[number].append((corresponding_l2_before, corresponding_l2))  # 숫자별로 (x, y) 저장

    return results

def extrack_plot_outtake(prey_deque, shared_mean):

    #prey 수의 집합 set
    num_set = set()
    for deque_values in prey_deque.values():
        num_set.update(deque_values)

    # 특정 prey수가 prey_deque 의 어떤 위치에 있는지 positions 에 저장
    pos_dict = {number: [] for number in num_set}

    # 각각의 prey 수가 담겨있는 위치가 어디인지 딕셔너리에 넣는 작업
    for agent_idx, deque_values in prey_deque.items():
        for idx, prey in enumerate(deque_values):
            pos_dict[prey].append((agent_idx, idx))

    # 결과를 저장할 딕셔너리
    results = []

    # pos_dict을 이용하여 결과 수집
    for prey, pos_list in pos_dict.items(): #pos_dict 에 있는 키와 쌍을 반환, 즉 number에는 숫자가 pop_list 에는 각 숫자가 있는 position 이 찍힌다.
        for pos in pos_list:
            x, y = pos #agent_idx 에는 position의 x축 값이, idx 에는 y 값이 찍힌다.
            corresponding_shared_mean = shared_mean[x][y]

            results.append((prey, corresponding_shared_mean))  # 숫자별로 (x, y) 저장

    return results



def from_semi_shared(semi_shared, pos, view_range):
    x_start = pos[1] + args.predator1_view_range -2
    y_start = pos[0] + args.predator1_view_range -2


    x_range = int(view_range)
    y_range = int(view_range)

    semi_shared = semi_shared[x_start - (x_range -1):x_start + (x_range+1), y_start - (y_range -1): y_start + (y_range+1)]

    return semi_shared


def to_semi_shared(semi_shared , pos, view_range):

    x_start = pos[1] + args.predator1_view_range - 2
    y_start = pos[0] + args.predator1_view_range - 2

    x_range = int(view_range)
    y_range = int(view_range)


    semi_shared[x_start - (x_range-1) :x_start + (x_range+1), y_start - (y_range - 1): y_start + (y_range+1)] += 1

    return semi_shared


def overlap_tiles(observation_temp):
    prey_pos = np.argwhere(observation_temp[:, :] >= 2)

    return prey_pos


#predator1 이 다른 predator1 과 predator2와 만나는 타일의 위치를 뽑는 함수
def coor_list_pred1(overlap_pos_list): #첫번째는 본인이고,그 뒤로 n_predator1 -1 개의 predator1, n_predator2개의 predator2

    #주인공 predator1 과 다른 predator1 들이 만나는 타일의 위치
    semi_shared_shape = (args.map_size + (predator1_view_range-2)*2, args.map_size + (predator1_view_range-2)*2)
    semi_shared = np.zeros(semi_shared_shape)

    for i in range(n_predator1-1): #처음 n_predator1 -1 개의 predator1에 대해 semi_shared 에 +1을 하여 기록한다.
        semi_shared = to_semi_shared(semi_shared , overlap_pos_list[i+1],predator1_view_range)

    with_pred1 = from_semi_shared(semi_shared,overlap_pos_list[0],predator1_view_range) #주인공 predator1가 보는 것을 가져온다.
    overlap_tiles_pred1 = np.argwhere(with_pred1[:, :] >= 1)#overlap_tiles 에 현재 주인공 predator1와 다른 predator1가 한번이라도 겹친 부분의 위치값들이 기록되어있다.


    # 주인공 predator1 과  predator2 들이 만나는 타일의 위치
    semi_shared = np.zeros(semi_shared_shape)

    for i in range(n_predator2):  # 처음 n_predator2 개의 predator2에 대해 semi_shared 에 +1을 하여 기록한다.
        semi_shared = to_semi_shared(semi_shared, overlap_pos_list[i + n_predator1], predator2_view_range)

    with_pred2 = from_semi_shared(semi_shared, overlap_pos_list[0], predator1_view_range)  # 주인공 predator1가 보는 것을 가져온다.
    overlap_tiles_pred2 = np.argwhere(with_pred2[:, :] >= 1)  # overlap_tiles 에 현재 주인공 predator1와 다른 predator2가 한번이라도 겹친 부분의 위치값들이 기록되어있다.

    return overlap_tiles_pred1, overlap_tiles_pred2


#predator1 이 다른 predator1 과 predator2와 만나는 타일의 위치를 뽑는 함수
def coor_list_pred2(overlap_pos_list): #첫번째는 predator2 본인이고,그 뒤로 n_predator1 개의 predator1, n_predator2 - 1 개의 predator2

    #주인공 predator2 과 다른 predator1 들이 만나는 타일의 위치
    semi_shared_shape = (args.map_size + (predator1_view_range-2)*2, args.map_size + (predator1_view_range-2)*2)
    semi_shared = np.zeros(semi_shared_shape)

    for i in range(n_predator1): #처음 n_predator1 -1 개의 predator1에 대해 semi_shared 에 +1을 하여 기록한다.
        semi_shared = to_semi_shared(semi_shared , overlap_pos_list[i+1],predator1_view_range)

    with_pred1 = from_semi_shared(semi_shared,overlap_pos_list[0],predator2_view_range) #주인공 predator2가 보는 것을 가져온다.
    overlap_tiles_pred1 = np.argwhere(with_pred1[:, :] >= 1)#overlap_tiles 에 현재 주인공 predator1와 다른 predator1가 한번이라도 겹친 부분의 위치값들이 기록되어있다.


    # 주인공 predator1 과  predator2 들이 만나는 타일의 위치
    semi_shared = np.zeros(semi_shared_shape) #초기화

    for i in range(n_predator2-1):  # 처음 n_predator2 개의 predator2에 대해 semi_shared 에 +1을 하여 기록한다.
        semi_shared = to_semi_shared(semi_shared, overlap_pos_list[i + n_predator1 + 1], predator2_view_range)

    with_pred2 = from_semi_shared(semi_shared, overlap_pos_list[0], predator2_view_range)  # 주인공 predator1가 보는 것을 가져온다.
    overlap_tiles_pred2 = np.argwhere(with_pred2[:, :] >= 1)  # overlap_tiles 에 현재 주인공 predator1와 다른 predator2가 한번이라도 겹친 부분의 위치값들이 기록되어있다.

    return overlap_tiles_pred1, overlap_tiles_pred2


def intake_sum(book, after_gnn, overlap_tiles):

    sum_of_squared_differences = 0

    for tiles in overlap_tiles: #tiles에는 각 픽셀의 좌표값들 들어있음
        # 각 위치에서 book과 after_gnn의 차이를 계산.
        difference = book[tiles[0], tiles[1]] - after_gnn[tiles[0], tiles[1]]

        sum_of_squared_differences += torch.sum(difference ** 2)


    return sum_of_squared_differences


def intake_inner(book, after_gnn, overlap_tiles):

    sum_of_inner_product = 0

    for tiles in overlap_tiles:  # tiles에는 각 픽셀의 좌표값들 들어있음
        # 각 위치에서 book과 after_gnn의 차이를 계산.
        mul = book[tiles[0], tiles[1]] * after_gnn[tiles[0], tiles[1]]

        sum_of_inner_product += torch.sum(mul)

    return sum_of_inner_product


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
        self.team_idx = None

        self.gdqn = None
        self.gdqn_target = None
        self.buffer = None

        self.gdqn_optimizer = None
        self.target_optimizer = None




        ##############################
        ######related to plotting#####
        ##############################

        #action 에 관한 것!
        self.ep_move_count_pred = {
            0: 1,
            1: 1
        }

        self.step_move_count_pred = {
            0: 1,
            1: 1
        }

        self.step_tag_count_pred = {
            0: 0,
            1: 0
        }

        # plotting 을 팀별로 찍기위해서 필요한 것!
        self.summation_team_dist = {0: np.array([], dtype=float), 1: np.array([], dtype=float)}

        self.avg_dist_deque_pred1 = deque(maxlen=args.deque_len)
        self.avg_dist_deque_pred2 = deque(maxlen=args.deque_len)

        self.min_dist_deque_pred1 = deque(maxlen=args.deque_len)
        self.min_dist_deque_pred2 = deque(maxlen=args.deque_len)

        self.avg_move_deque_pred1 = deque(maxlen=args.deque_len)
        self.avg_move_deque_pred2 = deque(maxlen=args.deque_len)

        self.avg_tag_deque_pred1 = deque(maxlen=args.deque_len)
        self.avg_tag_deque_pred2 = deque(maxlen=args.deque_len)


        # plotting 을 개인별로 찍기위해서 필요한 것!
        self.agent_min_dist_deque_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            self.agent_min_dist_deque_dict[agent_idx] = deque(maxlen=args.deque_len)

        self.agent_avg_dist_deque_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            self.agent_avg_dist_deque_dict[agent_idx] = deque(maxlen=args.deque_len)

        self.agent_action_deque_dict = {}  # 각 에이전트가 avg에 따라 액션(움직임,가만히있음,태그)을 어떻게 하는지 저장하기 위한 딕셔너리
        for agent_idx in range(n_predator1 + n_predator2):
            self.agent_action_deque_dict[agent_idx] = deque(maxlen=args.deque_len)


        #out take
        self.agent_graph_overlap_pred1_deque_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            self.agent_graph_overlap_pred1_deque_dict[agent_idx] = deque(maxlen=args.deque_len)

        self.agent_graph_overlap_pred2_deque_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            self.agent_graph_overlap_pred2_deque_dict[agent_idx] = deque(maxlen=args.deque_len)

        self.prey_number_deque_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            self.prey_number_deque_dict[agent_idx] = deque(maxlen=args.deque_len)


        self.shared_mean_deque_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            self.shared_mean_deque_dict[agent_idx] = deque(maxlen=args.deque_len)

        self.l2_before_outtake_deque_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            self.l2_before_outtake_deque_dict[agent_idx] = deque(maxlen=args.deque_len)


        self.l2_outtake_deque_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            self.l2_outtake_deque_dict[agent_idx] = deque(maxlen=args.deque_len)


        # in take

        self.l2_intake_deque_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            self.l2_intake_deque_dict[agent_idx] = deque(maxlen=args.deque_len)

        self.intake_overlap_with_pred1 = {}
        for agent_idx in range(n_predator1 + n_predator2):
            self.intake_overlap_with_pred1[agent_idx] = deque(maxlen=args.deque_len)

        self.intake_overlap_with_pred2 = {}
        for agent_idx in range(n_predator1 + n_predator2):
            self.intake_overlap_with_pred2[agent_idx] = deque(maxlen=args.deque_len)



        self.intake_sum_with_pred1_deque_dict = {} #수정한 Intake 담을 큐
        for agent_idx in range(n_predator1 + n_predator2):
            self.intake_sum_with_pred1_deque_dict[agent_idx] = deque(maxlen=args.deque_len)

        self.intake_sum_with_pred2_deque_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            self.intake_sum_with_pred2_deque_dict[agent_idx] = deque(maxlen=args.deque_len)

        self.intake_inner_with_pred1_deque_dict = {}  # 수정한 Intake 담을 큐
        for agent_idx in range(n_predator1 + n_predator2):
            self.intake_inner_with_pred1_deque_dict[agent_idx] = deque(maxlen=args.deque_len)

        self.intake_inner_with_pred2_deque_dict = {}  # 수정한 Intake 담을 큐
        for agent_idx in range(n_predator1 + n_predator2):
            self.intake_inner_with_pred2_deque_dict[agent_idx] = deque(maxlen=args.deque_len)

        self.tiles_number_with_pred1_deque_dict = {}  # 수정한 Intake 담을 큐
        for agent_idx in range(n_predator1 + n_predator2):
            self.tiles_number_with_pred1_deque_dict[agent_idx] = deque(maxlen=args.deque_len)

        self.tiles_number_with_pred2_deque_dict = {}  # 수정한 Intake 담을 큐
        for agent_idx in range(n_predator1 + n_predator2):
            self.tiles_number_with_pred2_deque_dict[agent_idx] = deque(maxlen=args.deque_len)

        #model
        self.gdqns = [G_DQN(self.dim_act, self.predator1_obs).to(self.device) for _ in range(self.n_predator1)] + [
            G_DQN(self.dim_act, self.predator2_obs).to(self.device) for _ in range(self.n_predator2)]

        self.gdqn_targets = [G_DQN(self.dim_act, self.predator1_obs).to(self.device) for _ in range(self.n_predator1)] + [
            G_DQN(self.dim_act, self.predator2_obs).to(self.device) for _ in range(self.n_predator2)]

        self.buffers = [ReplayBuffer(capacity=buffer_size) for _ in range(self.n_predator1 + self.n_predator2)]
        self.gdqn_optimizers = [Adam(x.parameters(), lr=args.lr) for x in self.gdqns]

        self.criterion = nn.MSELoss()



    #########################################
    ################setting##################
    #########################################

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

    def set_agent_buffer(self ,idx):
        self.buffer = self.buffers[idx]

    def set_agent_pos(self, pos):
        self.pos = pos

    def set_agent_shared(self, view_range):
        self.view_range = view_range

    def set_agent_model(self,agent):
        self.gdqn = self.gdqns[agent]
        self.gdqn_target = self.gdqn_targets[agent]



    #########################################
    ##############move & count###############
    #########################################

    # move count 에 대한 매서드
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

    # tag count 에 대한 매서드
    # for ep team move counting
    def step_tag_count(self):
        self.step_tag_count_pred[self.team_idx] += 1

    def reset_step_tag_count(self):
        self.step_tag_count_pred[0] = 0
        self.step_tag_count_pred[1] = 0



    ####################################
    ################dist################
    ####################################


    # 에이전트(=predator)와 prey들까지의 거리를 계산하는 함수
    def dist(self, obs):
        # prey position
        prey_pos = np.argwhere(obs[:,:,args.dim_feature-1] == 1)

        # predator position (본인 위치)
        pred_pos = np.array([[(self.view_range*2 - 1)/2, (self.view_range*2 - 1)/2]], dtype=np.int32)

        # entire position
        entire_pos = np.append(pred_pos, prey_pos ,axis=0)

        # calcualate distance
        distance_matrix = squareform(pdist(entire_pos, 'euclidean'))

        return distance_matrix[0,1:]

    def concat_dist(self, dist_list):



        self.summation_team_dist[self.team_idx] = np.concatenate((self.summation_team_dist[self.team_idx],dist_list))
        #self.summation_team_dist[self.team_idx].append(dist_list)
        print("a")

    def reset_summation_team_dist(self):
        self.summation_team_dist[0] = np.array([], dtype=float)
        self.summation_team_dist[1] = np.array([], dtype=float)


    def avg_dist(self, obs):
        # prey position
        prey_pos = np.argwhere(obs[:,:,args.dim_feature-1] == 1)

        # predator position (본인 위치)
        pred_pos = np.array([[(self.view_range*2 - 1)/2, (self.view_range*2 - 1)/2]], dtype=np.int32)

        # entire position
        entire_pos = np.append(pred_pos, prey_pos ,axis=0)

        # calcualate distance
        distance_matrix = squareform(pdist(entire_pos, 'euclidean'))

        distance_list = distance_matrix[0,1:]

        if distance_list.size > 0:
            # 배열이 비어 있지 않으므로 최소값 찾기
            avg_dist = np.mean(distance_list)
        else:
            avg_dist = self.view_range + 5

        return avg_dist


    def min_dist(self, obs):
        # prey position
        prey_pos = np.argwhere(obs[:, :, args.dim_feature-1] == 1)

        # predator position (본인위치)
        pred_pos = np.array([[(self.view_range * 2 - 1) / 2, (self.view_range * 2 - 1) / 2]], dtype=np.int32)

        # entire position
        entire_pos = np.append(pred_pos, prey_pos, axis=0)

        # calcualate distance
        distance_matrix = squareform(pdist(entire_pos, 'euclidean'))

        distance_list = distance_matrix[0, 1:]

        if distance_list.size > 0:
            # 배열이 비어 있지 않으므로 최소값 찾기
            min_dist = np.min(distance_list)
        else:
            min_dist = self.view_range + 5

        return min_dist



    #########################################
    ################guestbook################
    #########################################

    def from_guestbook(self):
        x_start = self.pos[1] + args.predator1_view_range -2
        y_start = self.pos[0] + args.predator1_view_range -2



        x_range = int(self.view_range)
        y_range = int(self.view_range)
        z_range = self.shared_shape[2]


        extracted_area = self.shared[x_start - (x_range -1):x_start + (x_range+1), y_start - (y_range -1): y_start + (y_range+1),
                         :]

        return extracted_area


    def to_guestbook(self, info):

        x_start = self.pos[1] + args.predator1_view_range - 2
        y_start = self.pos[0] + args.predator1_view_range - 2

        x_range = int(self.view_range)
        y_range = int(self.view_range)


        self.shared[x_start - (x_range-1) :x_start + (x_range+1), y_start - (y_range - 1): y_start + (y_range+1), :] += info


    def shared_decay(self):
        self.shared = self.shared * args.book_decay

    def reset_shred(self,shared):
        self.shared = shared



    def get_action(self, state, mask=None):

        # try:
        #     torch.cuda.empty_cache()
        # except:
        #     pass

        book = self.from_guestbook()
        q_value, shared_info, l2_before , l2_outtake, shared_sum, l2_intake , after_gnn = self.gdqn(torch.tensor(state).to(self.device), self.adj.to(self.device), book.to(self.device))

        #self.to_guestbook(shared_info.to('cpu'))
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)

        if np.random.random() < self.epsilon:
            print('random')
            # return random.randint(0, self.dim_act - 1), book
            return random.randint(0, dim_act-1), book, shared_info, l2_before,  l2_outtake, shared_sum, l2_intake ,after_gnn
        return torch.argmax(q_value).item(), book, shared_info, l2_before, l2_outtake, shared_sum, l2_intake, after_gnn

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

            q_values, _,_,_,_ = self.gdqn(observations.unsqueeze(0), adj.unsqueeze(0), book[0].detach().to(self.device))
            q_values = q_values[actions]


            next_q_values, _,_,_,_ = self.gdqn_target(next_observations.unsqueeze(0), adj.unsqueeze(0), book_next[0].detach().to(self.device))
            next_q_values = torch.max(next_q_values)

            targets = int(rewards[0]) + (1 - int(termination[0])) * next_q_values * args.gamma
            loss = self.criterion(q_values, targets.detach())
            loss.backward()

            self.gdqn_optimizer.step()


            try:
                torch.cuda.empty_cache()
            except:
                pass


    def target_update(self):
        weights = self.gdqn.state_dict()
        self.gdqn_target.load_state_dict(weights)




    #########################################
    ################plotting#################
    #########################################


    #avg_dist
    def avg_dist_append_pred1(self, dist):
        self.avg_dist_deque_pred1.append(dist)

    def avg_dist_append_pred2(self, dist):
        self.avg_dist_deque_pred2.append(dist)


    #min_dist
    def min_dist_append_pred1(self, dist):
        self.min_dist_deque_pred1.append(dist)


    def min_dist_append_pred2(self, dist):
        self.min_dist_deque_pred2.append(dist)

    #avg_move
    def avg_move_append_pred1(self, move):
        self.avg_move_deque_pred1.append(move)

    def avg_move_append_pred2(self, move):
        self.avg_move_deque_pred2.append(move)

    #avg_tag
    def avg_tag_append_pred1(self, tag):
        self.avg_tag_deque_pred1.append(tag)

    def avg_tag_append_pred2(self, tag):
        self.avg_tag_deque_pred2.append(tag)


    # 각 에이전트에 대한 append는 굳이 팀처럼 append 역할을 하는 함수를 따로 두지 않아도 된다.
    # def avg_dist_append_agent(self, avg_dist):
    #     self.agent_avg_dist_deque_dict[self.idx].append(avg_dist)

    # def min_dist_append_agent(self, min_dist):
    #     self.agent_min_dist_deque_dict[self.idx].append(min_dist)


    # def action_append_agent(self, min_dist):
    #     self.agent_action_deque_dict[self.idx].append(min_dist)


    def plot(self,ep):

        colors = [
            'blue', 'green', 'red', 'purple', 'orange', 'yellow',
            'lime', 'teal', 'cyan', 'magenta', 'pink', 'brown',
            'grey', 'navy', 'gold', 'crimson', 'violet', 'indigo',
            'tomato', 'turquoise', 'lavender', 'salmon', 'beige',
            'mint', 'coral', 'chocolate', 'maroon', 'olive',
            '#aaffc3', '#808000', '#ffd700', '#0048ba', '#b0bf1a',
            '#7cb9e8', '#c0e8d5', '#b284be', '#72a0c1', '#f0f8ff',
            '#e4717a', '#00ffff', '#a4c639', '#f4c2c2', '#915c83'
        ]

        ######################################################################################
        #################################out take 수정 이전#####################################
        ######################################################################################

        ################################
        ############case1###############
        ################################

        #전체
        # num_set = prey_num_set(self.prey_number_deque_dict)
        #
        # positions_dict = find_prey_positions(num_set, self.prey_number_deque_dict)
        #
        # results = extract_values_based_on_positions(num_set, positions_dict,self.l2_before_outtake_deque_dict, self.l2_outtake_deque_dict)
        #
        # # 결과를 prey 에 따라 plotting
        # plt.figure(figsize=(10, 6))
        #
        # for prey, coords in results.items():
        #     if coords:  # 좌표가 있는 경우에만
        #         x, y = zip(*coords)  # 숫자와 해당 값으로 분리
        #         plt.plot(x, y, 'o-', color=colors[prey], label=f'prey {prey}')  # 점과 선으로 연결
        #
        # plt.xlabel('Ot')
        # plt.ylabel('L2')
        # plt.title('out take_ case1')
        # plt.legend()
        #
        # wandb.log({"out take_ case1_ep_{}".format(ep): wandb.Image(plt)})
        #
        # plt.close()  # 현재 그림 닫기


        ####################
        ###predator1 team###
        ####################

        # prey_number_deque_dict_pred1 = {agent_idx: self.prey_number_deque_dict[agent_idx] for agent_idx in
        #                           range(n_predator1)}
        #
        # l2_before_outtake_deque_dict_pred1 = {agent_idx: self.l2_before_outtake_deque_dict[agent_idx] for agent_idx in
        #                                 range(n_predator1)}
        #
        # l2_outtake_deque_dict_pred1 = {agent_idx: self.l2_outtake_deque_dict[agent_idx] for agent_idx in
        #                                       range(n_predator1)}
        #
        #
        # num_set = prey_num_set(prey_number_deque_dict_pred1)
        # positions_dict = find_prey_positions(num_set, prey_number_deque_dict_pred1)
        # results = extract_values_based_on_positions(num_set, positions_dict, l2_before_outtake_deque_dict_pred1,
        #                                             l2_outtake_deque_dict_pred1)
        #
        # # 결과를 prey 에 따라 plotting
        # plt.figure(figsize=(10, 6))
        #
        # for prey, coords in results.items():
        #     if coords:  # 좌표가 있는 경우에만
        #         x, y = zip(*coords)  # 숫자와 해당 값으로 분리
        #         plt.plot(x, y, 'o-', color=colors[prey], label=f'prey {prey}')  # 점과 선으로 연결
        #
        # plt.xlabel('Ot')
        # plt.ylabel('L2')
        # plt.title('out take_pred1_case1')
        # plt.legend()
        #
        # wandb.log({"out take_pred1_case1_ep_{}".format(ep): wandb.Image(plt)})
        #
        # plt.close()  # 현재 그림 닫기

        ####################
        ###predator2 team###
        ####################

        # prey_number_deque_dict_pred2 = {agent_idx: self.prey_number_deque_dict[agent_idx] for agent_idx in
        #                                 range(n_predator1,n_predator1+n_predator2)}
        #
        # l2_before_outtake_deque_dict_pred2 = {agent_idx: self.l2_before_outtake_deque_dict[agent_idx] for agent_idx in
        #                                       range(n_predator1,n_predator1+n_predator2)}
        #
        # l2_outtake_deque_dict_pred2 = {agent_idx: self.l2_outtake_deque_dict[agent_idx] for agent_idx in
        #                                range(n_predator1,n_predator1+n_predator2)}
        #
        # num_set = prey_num_set(prey_number_deque_dict_pred2)
        # positions_dict = find_prey_positions(num_set, prey_number_deque_dict_pred2)
        # results = extract_values_based_on_positions(num_set, positions_dict, l2_before_outtake_deque_dict_pred2,
        #                                             l2_outtake_deque_dict_pred2)
        #
        # # 결과를 prey 에 따라 plotting
        # plt.figure(figsize=(10, 6))
        #
        # for prey, coords in results.items():
        #     if coords:  # 좌표가 있는 경우에만
        #         x, y = zip(*coords)  # 숫자와 해당 값으로 분리
        #         plt.plot(x, y, 'o-', color=colors[prey], label=f'prey {prey}')  # 점과 선으로 연결
        #
        # plt.xlabel('Ot')
        # plt.ylabel('L2')
        # plt.title('out take_pred2_case1')
        # plt.legend()
        #
        # wandb.log({"out take_pred2_case1_ep_{}".format(ep): wandb.Image(plt)})
        #
        # plt.close()  # 현재 그림 닫기

        ########################################################################################
        #################################out take 수정 version###################################
        ########################################################################################


        #predtor1 에 대한 정보만 가져옴
        prey_number_deque_dict_pred1 = {agent_idx: self.prey_number_deque_dict[agent_idx] for agent_idx in
                                        range(n_predator1)}
        prey_number_deque_dict_pred2 = {agent_idx: self.prey_number_deque_dict[agent_idx] for agent_idx in
                                        range(n_predator1,n_predator1+ n_predator2)}

        shared_mean_deque_dict_pred1 = {agent_idx: self.shared_mean_deque_dict[agent_idx] for agent_idx in
                                       range(n_predator1)}
        shared_mean_deque_dict_pred2 = {agent_idx: self.shared_mean_deque_dict[agent_idx] for agent_idx in
                                        range(n_predator1, n_predator1+n_predator2)}

        #predator1에 대해서 observation 안에 prey의 수의 집합 set을 만든다.
        results1 = extrack_plot_outtake(prey_number_deque_dict_pred1, shared_mean_deque_dict_pred1)
        results2 = extrack_plot_outtake(prey_number_deque_dict_pred2, shared_mean_deque_dict_pred2)
        # 결과를 prey 에 따라 plotting
        plt.figure(figsize=(10, 6))

        x1, y1 = zip(*results1)
        plt.scatter(x1, y1, facecolors='none', edgecolors=colors[0], label="predator1")

        x2, y2 = zip(*results2)
        plt.scatter(x2, y2, facecolors='none', edgecolors=colors[1], label="predator2")


        plt.xlabel('prey')
        plt.ylabel('shared_mean')
        plt.title('out take')
        plt.legend()
        # plt.show()

        wandb.log({"out take1_ep_{}".format(ep): wandb.Image(plt)})

        plt.close()  # 현재 그림 닫기

        ################################
        ############case2###############
        ################################

        # for agent in range(n_predator1 + n_predator2):
        #     plt.figure(figsize=(10, 6))
        #
        #     pred1_ratio_list = list(self.agent_graph_overlap_pred1_deque_dict[agent])
        #     pred2_ratio_list = list(self.agent_graph_overlap_pred2_deque_dict[agent])
        #     # 이게 아니라 l2 값인게 말이 된다.
        #
        #     l2 = list(self.l2_outtake_deque_dict[agent])
        #
        #     plt.scatter(pred1_ratio_list, l2, facecolors='none', edgecolors=colors[0], label="predator overlap")
        #     plt.scatter(pred2_ratio_list, l2, facecolors='none', edgecolors=colors[1], label="prey overlap")
        #
        #     plt.title('out take case1_predator1_{},ep_{}'.format(agent, ep))
        #     plt.xlabel('overlap')
        #     plt.ylabel('l2')
        #     plt.legend()
        #
        #     wandb.log({'out take case1_predator1_{},ep_{}'.format(agent, ep): wandb.Image(plt)})
        #
        #     plt.close()  # 현재 그림 닫기

        ##############################################################################
        ##################################in take#####################################
        ##############################################################################


        ################################
        ############case1###############
        ################################

        #각각의 에이전트에 대해서 그려본다.
        for agent in range(n_predator1+n_predator2):

            plt.figure(figsize=(10, 6))

            l2_intake= list(self.l2_intake_deque_dict[agent])
            overlap_pred1 = list(self.intake_overlap_with_pred1[agent])
            overlap_pred2 = list(self.intake_overlap_with_pred2[agent])

            plt.scatter(overlap_pred1 ,l2_intake, facecolors='none', edgecolors=colors[agent], label='about predator1')
            plt.scatter(overlap_pred2, l2_intake, facecolors='none', edgecolors=colors[agent+1], label='about predator2')


            plt.title("overlap ratio - l2_agent{}_ep_{}".format(agent,ep))
            plt.xlabel('overlap tiles')
            plt.ylabel('l2')
            plt.legend()

            wandb.log({"overlap ratio - l2_agent{}_ep_{}".format(agent,ep): wandb.Image(plt)})

            plt.close()  # 현재 그림 닫기

            # 수정한 intake
            # intake_sum_with_pred1 = intake_sum(book, after_gnn, overlap_tiles_pred1)
            # intake_sum_with_pred2 = intake_sum(book, after_gnn, overlap_tiles_pred2)
            # madqn.intake_sum_with_pred1_deque_dict[idx].append(intake_sum_with_pred1)
            # madqn.intake_sum_with_pred2_deque_dict[idx].append(intake_sum_with_pred2)
            #
            # intake_inner_with_pred1 = intake_inner(book, after_gnn, overlap_tiles_pred1)
            # intake_inner_with_pred2 = intake_inner(book, after_gnn, overlap_tiles_pred2)
            # madqn.intake_inner_with_pred1_deque_dict[idx].append(intake_inner_with_pred1)
            # madqn.intake_inner_with_pred2_deque_dict[idx].append(intake_inner_with_pred2)
            #
            # madqn.tiles_number_with_pred1_deque_dict[idx].append(len(overlap_tiles_pred1))
            # madqn.tiles_number_with_pred2_deque_dict[idx].append(len(overlap_tiles_pred2))

        for agent in range(n_predator1 + n_predator2):

            tiles_num_with_pred1 = list(self.tiles_number_with_pred1_deque_dict[agent])
            tiles_num_with_pred2 = list(self.tiles_number_with_pred2_deque_dict[agent])

            intake_sum_with_pred1 = list(self.intake_sum_with_pred1_deque_dict[agent])
            intake_sum_with_pred2 = list(self.intake_sum_with_pred2_deque_dict[agent])

            intake_inner_with_pred1 = list(self.intake_inner_with_pred1_deque_dict[agent])
            intake_inner_with_pred2 = list(self.intake_inner_with_pred2_deque_dict[agent])

            plt.figure(figsize=(10, 6))

            plt.scatter(tiles_num_with_pred1, intake_sum_with_pred1, facecolors='none', edgecolors=colors[agent],
                        label='about predator1')
            plt.scatter(tiles_num_with_pred2, intake_sum_with_pred2, facecolors='none', edgecolors=colors[agent],
                        label='about predator1')

            plt.title("intake_sum_agent{}_ep_{}".format(agent, ep))
            plt.xlabel('overlap tiles number')
            plt.ylabel('sum of squared differences')
            plt.legend()

            wandb.log({"intake_sum_agent{}_ep_{}".format(agent, ep): wandb.Image(plt)})

            plt.close()  # 현재 그림 닫기

            plt.figure(figsize=(10, 6))

            plt.scatter(tiles_num_with_pred1, intake_inner_with_pred1, facecolors='none', edgecolors=colors[agent],
                        label='about predator1')
            plt.scatter(tiles_num_with_pred2, intake_inner_with_pred2, facecolors='none', edgecolors=colors[agent],
                        label='about predator1')

            plt.title("intake_inner_agent{}_ep_{}".format(agent, ep))
            plt.xlabel('overlap tiles number')
            plt.ylabel('inner product')
            plt.legend()

            wandb.log({"intake_inner_agent{}_ep_{}".format(agent, ep): wandb.Image(plt)})

            plt.close()  # 현재 그림 닫기

        ################################
        ############case2###############
        ################################


        ##############################################################################
        ###############################phenomenon#####################################
        ##############################################################################


        ############################################################
        ########################for each team#######################
        ############################################################

        jitter_std = args.jitter_std

        avg_dist_list_pred1 = list(self.avg_dist_deque_pred1)
        avg_dist_list_pred2 = list(self.avg_dist_deque_pred2)

        min_dist_list_pred1 = list(self.min_dist_deque_pred1)
        min_dist_list_pred2 = list(self.min_dist_deque_pred2)

        avg_move_list_pred1 = list(self.avg_move_deque_pred1)
        avg_move_list_pred2 = list(self.avg_move_deque_pred2)

        avg_tag_list_pred1 = list(self.avg_tag_deque_pred1)
        avg_tag_list_pred2 = list(self.avg_tag_deque_pred2)

        ###################################
        #### avg(distance) - avg(move)#####
        ###################################

        plt.figure(figsize=(10, 6))

        plt.scatter(avg_dist_list_pred1 + np.random.normal(0, jitter_std, len(avg_dist_list_pred1)),
                    avg_move_list_pred1, facecolors='none', edgecolors='blue', label='pred1')
        plt.scatter(avg_dist_list_pred2 + np.random.normal(0, jitter_std, len(avg_dist_list_pred2)),
                    avg_move_list_pred2, facecolors='none', edgecolors='red', label='pred2')

        plt.title('avg dist : avg_move')
        plt.xlabel('avg_dist')
        plt.ylabel('avg_move')
        plt.legend()


        wandb.log({"avg dist : avg_move_ep_{}".format(ep): wandb.Image(plt)})

        plt.close()  # 현재 그림 닫기

        ###################################
        #### min(distance) - avg(move)#####
        ###################################

        plt.figure(figsize=(10, 6))
        plt.scatter(min_dist_list_pred1 + np.random.normal(0, jitter_std, len(min_dist_list_pred1)),
                    avg_move_list_pred1, facecolors='none', edgecolors='blue', label='pred1')
        plt.scatter(min_dist_list_pred2 + np.random.normal(0, jitter_std, len(min_dist_list_pred2)),
                    avg_move_list_pred2, facecolors='none', edgecolors='red', label='pred2')

        plt.title('min move for the avg distance')
        plt.xlabel('min_dist')
        plt.ylabel('avg_move')
        plt.legend()


        wandb.log({"min move for the avg distance_ep_{}".format(ep): wandb.Image(plt)})

        plt.close()  # 현재 그림 닫기

        ###################################
        #### min(distance) - avg(tag)######
        ###################################

        plt.figure(figsize=(10, 6))
        plt.scatter(min_dist_list_pred1 + np.random.normal(0, jitter_std, len(min_dist_list_pred1)),
                    avg_tag_list_pred1, facecolors='none', edgecolors='blue', label='pred1')
        plt.scatter(min_dist_list_pred2 + np.random.normal(0, jitter_std, len(min_dist_list_pred2)),
                    avg_tag_list_pred2, facecolors='none', edgecolors='red', label='pred2')

        plt.title('min_dist : avg_tag')
        plt.xlabel('min_dist')
        plt.ylabel('avg_tag')
        plt.legend()

        wandb.log({"min_dist : avg_tag_ep_{}".format(ep): wandb.Image(plt)})

        plt.close()  # 현재 그림 닫기

        ###################################
        #### avg(distance) - avg(tag)######
        ###################################


        plt.figure(figsize=(10, 6))
        plt.scatter(avg_dist_list_pred1 + np.random.normal(0, jitter_std, len(min_dist_list_pred1)),
                    avg_tag_list_pred1, facecolors='none', edgecolors='blue', label='pred1')
        plt.scatter(avg_dist_list_pred2 + np.random.normal(0, jitter_std, len(min_dist_list_pred2)),
                    avg_tag_list_pred2, facecolors='none', edgecolors='red', label='pred2')

        plt.title('avg_dist : avg_tag')
        plt.xlabel('avg_dist')
        plt.ylabel('avg_tag')
        plt.legend()

        wandb.log({"avg_dist : avg_tag_ep_{}".format(ep): wandb.Image(plt)})

        plt.close()  # 현재 그림 닫기

        ###################################################################
        ###########################for one agent###########################
        ###################################################################

        ################################
        #### min(distance) - action#####
        ################################
        #predator1

        plt.figure(figsize=(10, 6))

        for agent in range(n_predator1):

            min_dist_list = list(self.agent_min_dist_deque_dict[agent])
            action_list = list(self.agent_action_deque_dict[agent])

            plt.scatter(min_dist_list + np.random.normal(0, jitter_std, len(avg_dist_list_pred1)),
                        action_list, facecolors='none', edgecolors=colors[agent], label='agent_{}'.format(agent))

        plt.title('pred1 ->min dist : action')
        plt.xlabel('min dist')
        plt.ylabel('action')
        plt.legend()

        wandb.log({"pred1_agent_min dist : action_ep_{}".format(ep): wandb.Image(plt)})

        plt.close()  # 현재 그림 닫기

        # predator2
        plt.figure(figsize=(10, 6))

        for agent in range(n_predator2):
            min_dist_list = list(self.agent_min_dist_deque_dict[n_predator1+agent])
            action_list = list(self.agent_action_deque_dict[n_predator1+agent])

            plt.scatter(min_dist_list + np.random.normal(0, jitter_std, len(avg_dist_list_pred1)),
                        action_list, facecolors='none', edgecolors=colors[agent], label='agent_{}'.format(agent+n_predator1))

        plt.title('pred2 -> min dist : action')
        plt.xlabel('min dist')
        plt.ylabel('action')
        plt.legend()

        wandb.log({"pred2_agent_min dist : action_ep_{}".format(ep): wandb.Image(plt)})

        plt.close()  # 현재 그림 닫기

        ################################
        #### avg(distance) - action#####
        ################################
        # predator1

        plt.figure(figsize=(10, 6))

        for agent in range(n_predator1):
            avg_dist_list = list(self.agent_avg_dist_deque_dict[agent])
            action_list = list(self.agent_action_deque_dict[agent])

            plt.scatter(avg_dist_list + np.random.normal(0, jitter_std, len(avg_dist_list)),
                        action_list, facecolors='none', edgecolors=colors[agent], label='agent_{}'.format(agent))

        plt.title('pred1 ->avg dist : action')
        plt.xlabel('avg dist')
        plt.ylabel('action')
        plt.legend()

        wandb.log({"pred1_agent_avg dist : action_ep_{}".format(ep): wandb.Image(plt)})

        plt.close()  # 현재 그림 닫기

        # predator2

        plt.figure(figsize=(10, 6))

        for agent in range(n_predator2):
            avg_dist_list = list(self.agent_avg_dist_deque_dict[n_predator1 + agent])
            action_list = list(self.agent_action_deque_dict[n_predator1 + agent])

            plt.scatter(avg_dist_list + np.random.normal(0, jitter_std, len(avg_dist_list)),
                        action_list, facecolors='none', edgecolors=colors[agent],
                        label='agent_{}'.format(agent + n_predator1))

        plt.title('pred2 -> avg dist : action')
        plt.xlabel('avg dist')
        plt.ylabel('action')
        plt.legend()

        wandb.log({"pred2_agent_avg dist : action_ep_{}".format(ep): wandb.Image(plt)})

        plt.close()  # 현재 그림 닫기

from magent2.environments import hetero_adversarial_v1
from magent2.environments import hetero_adversarial_v1
from MADQN import MADQN, calculate_Overlap_ratio, coor_list_pred1, coor_list_pred2, intake_sum, intake_inner, \
    prey_number, calculate_Overlap_ratio_intake

import numpy as np
import torch as th
import wandb

from arguments import args

wandb.init(project="MADQN", entity='hails',config=args.__dict__)
wandb.run.name = 'analysis_mac'


device = 'cpu'

render_mode = 'rgb_array'
# render_mode = 'human'

predator1_view_range = args.predator1_view_range
predator2_view_range = args.predator2_view_range
n_predator1 = args.n_predator1
n_predator2 = args.n_predator2
n_prey = args.n_prey
dim_feature = args.dim_feature

shared_shape = (
args.map_size + (predator1_view_range - 2) * 2, args.map_size + (predator1_view_range - 2) * 2, dim_feature)
predator1_obs = (predator1_view_range * 2, predator1_view_range * 2, dim_feature)
predator2_obs = (predator2_view_range * 2, predator2_view_range * 2, dim_feature)
dim_act = 13

predator1_adj = ((predator1_view_range * 2) ** 2, (predator1_view_range * 2) ** 2)
predator2_adj = ((predator2_view_range * 2) ** 2, (predator2_view_range * 2) ** 2)

batch_size = 1

shared = th.zeros(shared_shape)
madqn = MADQN(n_predator1, n_predator2, predator1_obs, predator2_obs, dim_act, shared_shape, shared, args.buffer_size,
              device)


# for i in range(n_predator1 + n_predator2):
# 	madqn.gdqns[i].load_state_dict(th.load(f'./model_cen_save/model_{i}_ep50.pt'))

# model_file_name = f'./model_cen_save/model_{i}_ep50.pt'
# model_state_dict = th.load(model_file_name)
# madqn.gdqns[i].load_state_dict(model_state_dict)


# def process_array_1(arr):  #predator1 (obs, team, team_hp, predator2, predator2 hp, prey, prey hp)
#     arr = np.delete(arr, [2, 4, 6], axis=2)
#     combined_dim = np.logical_or(arr[:, :, 1], arr[:, :, 2])
#     result = np.dstack((arr[:, :, 0], combined_dim, arr[:, :, 3]))
#
#     return result
#
#
# def process_array_2(arr): #predator2 (obs, team, team_hp, prey, prey hp, predator2, predator2 hp)
#     arr = np.delete(arr, [2, 4, 6], axis=2)
#     combined_dim = np.logical_or(arr[:, :, 1], arr[:, :, 3])
#     result = np.dstack((arr[:, :, 0], combined_dim, arr[:, :, 2]))
#
#     return result

def process_array_1(arr):  # predator1 (obs, team, team_hp, predator2, predator2 hp, prey, prey hp)

    arr = np.delete(arr, [2, 4, 6], axis=2)
    result = np.dstack((arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]))

    return result


def process_array_2(arr):  # predator2 (obs, team, team_hp, prey, prey hp, predator2, predator2 hp)

    arr = np.delete(arr, [2, 4, 6], axis=2)
    result = np.dstack((arr[:, :, 0], arr[:, :, 1], arr[:, :, 3], arr[:, :, 2]))

    return result


# ?? 4?? ??? ??? observation?? prey? ?? ?? ?, min,mean,max??? ??? ? ? ???? ??? ??? ?? ????.
def check_zero_size_min_pred1(list):
    if list.size > 0:
        # ??? ?? ?? ???? ??? ??
        min_value = np.min(list)
    else:
        min_value = args.predator1_view_range + 1

    return min_value


def check_zero_size_min_pred2(list):
    if list.size > 0:
        # ??? ?? ?? ???? ??? ??
        min_value = np.min(list)
    else:
        min_value = args.predator2_view_range + 1

    return min_value


def check_zero_size_avg_pred1(list):
    if list.size > 0:
        # ??? ?? ?? ???? ??? ??
        avg_value = np.mean(list)
    else:
        avg_value = args.predator1_view_range + 1

    return avg_value


def check_zero_size_avg_pred2(list):
    if list.size > 0:
        # ??? ?? ?? ???? ??? ??
        avg_value = np.mean(list)
    else:
        avg_value = args.predator2_view_range + 1

    return avg_value


def main():
    env = hetero_adversarial_v1.env(map_size=args.map_size, minimap_mode=False, tag_penalty=args.tag_penalty,
                                    max_cycles=args.max_update_steps, extra_features=False, render_mode=render_mode)

    for ep in range(args.total_ep):

        # shared book reset every episode
        madqn.reset_shred(shared)

        # reset ep_move_count : ? ?????? Plot? ??? ?? ?? action? ???? ????? ???? ??
        madqn.reset_ep_move_count()

        # env reset
        env.reset(seed=args.seed)

        ep_reward = 0
        ep_reward_pred1 = 0
        ep_reward_pred2 = 0

        iteration_number = 0

        observations_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            observations_dict[agent_idx] = []

        reward_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            reward_dict[agent_idx] = []

        move_penalty_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            move_penalty_dict[agent_idx] = []

        action_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            action_dict[agent_idx] = []

        termination_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            termination_dict[agent_idx] = []

        truncation_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            truncation_dict[agent_idx] = []

        book_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            book_dict[agent_idx] = []

        shared_info_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            shared_info_dict[agent_idx] = []

        agent_pos = {}
        for agent_idx in range(n_predator1 + n_predator2):
            agent_pos[agent_idx] = []

        entire_pos = []

        print("ep:", ep, '*' * 80)

        for agent in env.agent_iter():

            step_idx = iteration_number // (args.n_predator1 + args.n_predator2 + args.n_prey)


            if (((iteration_number) % (args.n_predator1 + args.n_predator2 + args.n_prey)) == 0) and (step_idx > 0):

                ##########################################################################################
                ########### ? ??? ??? ??? step ?  ?? ??? shared graph ? ???? ???? ??##########
                ##########################################################################################
                ##??## max_update_steps ? ???  env.last()? ? step ? ??? truncation=TRUE ? ?? ?? ????? ??? ??? ??!!

                if step_idx > 0:

                    madqn.shared_decay() # ? step ? ??? decaying ? ???.

                    # ??? ??? ????, truncation ?? ?? ??? ??? ?????, ??? step? ?? ?? ???? shared_book? ???? ??? ?? ??.
                    if step_idx != args.max_update_steps:

                        # ??? ?? ????? share_book ? decaying??? ??? ??? ??.
                        if step_idx <= args.book_term:

                            # ?? ?????? ??? ??? ??
                            for idx in range(n_predator1 + n_predator2):

                                # ?? ???? agent_pos? self.pos? ??
                                madqn.set_agent_pos(agent_pos[idx][-1])

                                # predator ??? ?? self.view_range ??
                                if idx < args.n_predator1:
                                    madqn.set_agent_shared(predator1_view_range)
                                else:
                                    madqn.set_agent_shared(predator2_view_range)

                                # ????? ?? ??? shared_info? ??
                                madqn.to_guestbook(shared_info_dict[idx][-1].to('cpu'))

                        # step_idx > args.book_term ??? ?? ??? ??? ?????? ?? ??? ????? ??.
                        else:
                            # erase last step book information
                            for idx in range(n_predator1 + n_predator2):

                                madqn.set_agent_pos(agent_pos[idx][-(args.book_term + 1)])

                                if idx < args.n_predator1:
                                    madqn.set_agent_shared(predator1_view_range)
                                else:
                                    madqn.set_agent_shared(predator2_view_range)

                                # self.to_guestbook(shared_info.to('cpu'))
                                madqn.to_guestbook(-(args.book_decay ** (args.book_term)) * shared_info_dict[idx][
                                    -(args.book_term + 1)].to('cpu'))

                            # Add recent Step information
                            for idx in range(n_predator1 + n_predator2):

                                madqn.set_agent_pos(agent_pos[idx][-1])

                                if idx < args.n_predator1:
                                    madqn.set_agent_shared(predator1_view_range)
                                else:
                                    madqn.set_agent_shared(predator2_view_range)

                                madqn.to_guestbook(shared_info_dict[idx][-1].to('cpu'))

                        ######################################################################
                        ######################################################################
                        ######### avg(dist(prey))-move plotting process for each team#########
                        ######################################################################
                        ######################################################################
                        # ??? ??? ??(== truncation ?? ?? ???? ???? ??)
                        # ?? ?? ???? ??? ???? ??? ? ???? ???? ??

                        # madqn.summation_team_dist ??  step? ?? prey??? ???? ??? list
                        # ???? -1? ??? reward/move count ?? move count = 0 ?? ??? ?? ??? ???? ??? ???? ?? ??? 1? ?????
                        # ???????, ???? ?? ??? ??? ??? ?? ??? -1? ??? ???.

                        # # ??? ?? ? step??? ? predator1 ? prey???? ??? avg(distance) ,min(distance) ? avg(count)??? ??? ??
                        madqn.avg_dist_append_pred1(check_zero_size_avg_pred1(madqn.summation_team_dist[0]))  # step
                        madqn.min_dist_append_pred1(check_zero_size_min_pred1(madqn.summation_team_dist[0]))  # step
                        madqn.avg_move_append_pred1((madqn.step_move_count_pred[0] - 1) / n_predator1)  # step
                        madqn.avg_tag_append_pred1((madqn.step_tag_count_pred[0]) / n_predator1)  # step

                        # # ??? ?? ? step??? ? predator2 ? prey???? ??? avg(distance) ,min(distance) ? avg(count)??? ??? ??
                        madqn.avg_dist_append_pred2(check_zero_size_avg_pred2(madqn.summation_team_dist[1]))  # step
                        madqn.min_dist_append_pred2(check_zero_size_min_pred2(madqn.summation_team_dist[1]))  # step
                        madqn.avg_move_append_pred2((madqn.step_move_count_pred[1] - 1) / n_predator2)  # step
                        madqn.avg_tag_append_pred2((madqn.step_tag_count_pred[1]) / n_predator2)  # step

                        # # ??? ??? ??????? ?? ??? ???? plotting ? ?? ??? ?? ??->?? ??
                        madqn.reset_step_move_count()  # ????? ?? ????? ??.
                        madqn.reset_step_tag_count()  # ????? ?? ????? ??.
                        madqn.reset_summation_team_dist()  # ??? ????? ??.

                        # out take? ??? plot(==case2)? ?? ?? ??? ??? ???? ??
                        pos_list = np.concatenate((pos_predator1, pos_predator2), axis=0)
                        # ? ?????? preator1? predator2? ??? ??? ??? ??? ?? -> ???: predator1? ??? ?? , ????: predator2? ??? ??
                        ratio_matrix = calculate_Overlap_ratio(pos_list)

                        #? ????? ?? Observation ?? ?? predator1? ??? ??
                        for idx, value in enumerate(ratio_matrix[:, 0]):
                            madqn.agent_graph_overlap_pred1_deque_dict[idx].append(value)

                        #? ????? ?? Observation ?? ?? predator2? ??? ??
                        # a? ? ?? ?? ??? self.agent_graph_overlap_pred2_deque_dict? ??
                        for idx, value in enumerate(ratio_matrix[:, 1]):
                            madqn.agent_graph_overlap_pred2_deque_dict[idx].append(value)




                    # ??? step, ? truncation == 1 ? ???? agent? ???? ????, ??? ???? ???? ??? ????.
                    # ??? ??? ??? ??? ???? ??? ? ?? ??? ??.
                    else:
                        madqn.avg_dist_append_pred1(check_zero_size_avg_pred1(madqn.summation_team_dist[0]))
                        madqn.min_dist_append_pred1(check_zero_size_min_pred1(madqn.summation_team_dist[0]))
                        madqn.avg_move_append_pred1((madqn.step_move_count_pred[0] - 1) / n_predator1)
                        madqn.avg_tag_append_pred1((madqn.step_tag_count_pred[0]) / n_predator1)

                        # ??? ?? predator2 ? avg(distance) ,min(distance) ? avg(count)??? ??? ??
                        madqn.avg_dist_append_pred2(check_zero_size_avg_pred2(madqn.summation_team_dist[1]))
                        madqn.min_dist_append_pred2(check_zero_size_min_pred2(madqn.summation_team_dist[1]))
                        madqn.avg_move_append_pred2((madqn.step_move_count_pred[1] - 1) / n_predator2)
                        madqn.avg_tag_append_pred2((madqn.step_tag_count_pred[1]) / n_predator2)

                        # ??? ??? ??????? ?? ??? ???? plotting ? ?? ??? ?? ??->?? ??
                        madqn.reset_step_move_count()  # ????? ?? ????? ??.
                        madqn.reset_step_tag_count()  # ????? ?? ????? ??.
                        madqn.reset_summation_team_dist()  # ??? ????? ??.

                        # out take case2 ????? ? ??
                        # pos_predator1 ? pos_predator2 ? ?? ???? ?????, ??? ?? ?? ???, ???? ???? ????.
                        pos_list = np.concatenate((pos_predator1, pos_predator2), axis=0)
                        ratio_matrix = calculate_Overlap_ratio(pos_list)
                        # ? ??? ??? ??.
                        for idx, value in enumerate(ratio_matrix[:, 0]):
                            madqn.agent_graph_overlap_pred1_deque_dict[idx].append(value)

                        # a? ? ?? ?? ??? self.agent_graph_overlap_pred2_deque_dict? ??
                        for idx, value in enumerate(ratio_matrix[:, 1]):
                            madqn.agent_graph_overlap_pred2_deque_dict[idx].append(value)



                elif step_idx > 1:

                    ##########################################################################################
                    ############ ? ?? ??? ??? ??? step ?  ?? ??? buffer ? ???? ???? ??###########
                    ##########################################################################################
                    # put experience into the buffer after second step


                    # ?? ????? ?? reward ??? ?? ??
                    step_rewards = 0
                    step_penalty_rewards = 0

                    # ? ??? reward? ???? ?? ??
                    step_reward_pred1 = 0
                    step_reward_pred2 = 0
                    step_penalty_pred1 = 0
                    step_penalty_pred2 = 0
                    step_rewards_pred1 = 0
                    step_rewards_pred2 = 0

                    ##########################################################################
                    # ?? ????? total reward ?? -> ?????? ?? reward ? ???? ???? ???? ??
                    ##########################################################################

                    # ?????? ?? reward ??
                    for agent_rewards in reward_dict.values():
                        step_rewards += np.sum(agent_rewards[-1])

                    # ??? ??? ??
                    for penalty in move_penalty_dict.values():
                        step_penalty_rewards += np.sum(penalty[-2])

                    # ?? ? ??? ??? ? ???? ?? reward? ???? ??
                    step_rewards = step_rewards + step_penalty_rewards

                    # ep ? ?? ?? reward? ????? ??
                    ep_reward += step_rewards

                    ##########################################################################
                    # ???? reward ?? -> ?????? ?? reward ? ???? ???? ???? ??
                    ##########################################################################

                    # ? ??? ??????? reward ??
                    for i, agent_rewards in enumerate(reward_dict.values()):
                        if i < len(reward_dict) // 2:
                            step_reward_pred1 += np.sum(agent_rewards[-1])
                        else:
                            step_reward_pred2 += np.sum(agent_rewards[-1])

                    # ? ??? ??????? reward ??
                    for i, agent_penalty in enumerate(move_penalty_dict.values()):
                        if i < len(reward_dict) // 2:
                            step_penalty_pred1 += np.sum(agent_penalty[-1])
                        else:
                            step_penalty_pred2 += np.sum(agent_penalty[-1])

                    step_rewards_pred1 = step_reward_pred1 + step_penalty_pred1
                    step_rewards_pred2 = step_reward_pred2 + step_penalty_pred2

                    ep_reward_pred1 += step_rewards_pred1
                    ep_reward_pred2 += step_rewards_pred2

                    for idx in range(n_predator1 + n_predator2):
                        madqn.set_agent_buffer(idx)  # ? ????? ??? buffer ? ????? ??

                        madqn.buffer.put(observations_dict[idx][-2],
                                         book_dict[idx][-2],
                                         action_dict[idx][-2],
                                         step_rewards,
                                         observations_dict[idx][-1],
                                         book_dict[idx][-1],
                                         termination_dict[idx][-2],
                                         truncation_dict[idx][-2])




                    #########################################################################
                    ################################wandb ??################################
                    #########################################################################
                    # ? step ? ???, ? step?? ??? ? ???? wandb ? ???? ??!

                    ############################################
                    #####################??####################
                    ############################################


                    wandb.log({"Total Step Reward":  step_rewards, "Team1 Step Reward": step_rewards_pred1, "Team2 Step Reward": step_rewards_pred2})


                    wandb.log({"Team 1 move count": madqn.step_move_count_pred[0][-1],
                               "Team 2 move count": madqn.step_move_count_pred[1][-1],
                               "Team 1 tag count": madqn.step_tag_count_pred[0][-1],
                               "Team 2 tag count": madqn.step_tag_count_pred[1][-1]})

                    wandb.log({"Team 1 avg tag count": madqn.avg_tag_deque_pred1[-1],
                               "Team 2 avg tag count": madqn.avg_tag_deque_pred2[-1],
                               "Team 1 avg move count": madqn.avg_move_deque_pred1[-1],
                               "Team 2 avg move count": madqn.avg_move_deque_pred2[-1]})







                    ############################################
                    ####################???####################
                    ############################################


                    #########################
                    #######Y?? ?? ??######
                    #########################

                    # ? ????? ?? Observation ?? ?? predator1? ??? ??
                    for idx, value in enumerate(madqn.agent_graph_overlap_pred1_deque_dict):
                        # ? deque? ??? ?? ??
                        last_value = value[-1]
                        wandb.log({f"Overlap Tile Ratio with Team 1_{idx}": last_value, "step": iteration_number})

                    # ? ????? ?? Observation ?? ?? predator2? ??? ??
                    for idx, value in enumerate(madqn.agent_graph_overlap_pred2_deque_dict):
                        # ? deque? ??? ?? ??
                        last_value = value[-1]
                        wandb.log({f"Overlap Tile Ratio with Team 2_{idx}": last_value, "step": iteration_number})

                    # team1? view? ??? ??? intake? ??? ????? ????? ??
                    for idx, value in enumerate(madqn.intake_sum_with_pred1_deque_dict):
                        # ? deque? ??? ?? ??
                        last_value = value[-1]
                        wandb.log({f"summed intake Difference with Team1_{idx}": last_value, "step": iteration_number})

                    # team2? view? ??? ??? intake? ??? ????? ????? ??
                    for idx, value in enumerate(madqn.intake_sum_with_pred2_deque_dict):
                        # ? deque? ??? ?? ??
                        last_value = value[-1]
                        wandb.log({f"summed intake Difference with Team2_{idx}": last_value, "step": iteration_number})

                    # team1? view? ??? ??? intake? ??? ????? ??? ???? ??
                    for idx, value in enumerate(madqn.intake_inner_with_pred1_deque_dict):
                        # ? deque? ??? ?? ??
                        last_value = value[-1]
                        wandb.log({f"inner product intake Difference with Team1_{idx}": last_value, "step": iteration_number})

                    # team2? view? ??? ??? intake? ??? ????? ??? ???? ??
                    for idx, value in enumerate(madqn.intake_inner_with_pred2_deque_dict):
                        # ? deque? ??? ?? ??
                        last_value = value[-1]
                        wandb.log({f"inner product intake Difference with Team2_{idx}": last_value, "step": iteration_number})

                    # # ??? observation ? GNN ? ?? ???? ?? observation ? L2 ?
                    # for idx, value in enumerate(madqn.l2_before_outtake_deque_dict):
                    #     # ? deque? ??? ?? ??
                    #     last_value = value[-1]
                    #     wandb.log(
                    #         {f"Total before outtake_{idx}": last_value, "step": iteration_number})

                    # ?? ?? ??? view ? ???? ?? ?? outtake ??? ???
                    for idx, value in enumerate(madqn.l2_outtake_deque_dict):
                        # ? deque? ??? ?? ??
                        last_value = value[-1]
                        wandb.log(
                            {f"Total outtake_{idx}": last_value, "step": iteration_number})

                    # ?? ?? ??? view ? ???? ?? ?? intake ??? ???
                    for idx, value in enumerate(madqn.l2_intake_deque_dict):
                        # ? deque? ??? ?? ??
                        last_value = value[-1]
                        wandb.log(
                            {f"Total Intake_{idx}": last_value, "step": iteration_number})

                    # ?  ????? action
                    for idx, value in enumerate(madqn.agent_action_deque_dict[idx]):
                        # ? deque? ??? ?? ??
                        last_value = value[-1]
                        wandb.log(
                            {f"Action_{idx}": last_value, "step": iteration_number})

                    #########################
                    #######Y?? ?? ??######
                    #########################

                    # ?? ??? ??? ??? ??? ???? ?? ??? ??? ???, ??? ??? ?? ????? ??? ?? '??? ?'? ???? ?
                    # team1 ? ??? ?? ?
                    for idx, value in enumerate(madqn.tiles_number_with_pred1_deque_dict):
                        # ? deque? ??? ?? ??
                        last_value = value[-1]
                        wandb.log( {f"Number of Overlapping Tiles with Team1_{idx}": last_value, "step": iteration_number})

                    # team2 ? ??? ?? ?
                    for idx, value in enumerate(madqn.tiles_number_with_pred2_deque_dict):
                    # ? deque? ??? ?? ??
                        last_value = value[-1]
                        wandb.log({f"Number of Overlapping Tiles with Team2_{idx}": last_value, "step": iteration_number})

                    # prey??? ?? ??
                    for idx, value in enumerate(madqn.agent_avg_dist_deque_dict):
                        # ? deque? ??? ?? ??
                        last_value = value[-1]
                        wandb.log(
                            {f"Average Distance to Prey_{idx}": last_value, "step": iteration_number})

                    # prey??? ?? ??
                    for idx, value in enumerate(madqn.agent_min_dist_deque_dict):
                        # ? deque? ??? ?? ??
                        last_value = value[-1]
                        wandb.log(
                            {f"Shortest Distance to Prey_{idx}": last_value, "step": iteration_number})

                    # ?  ???? ?? ???? prey ?
                    for idx, value in enumerate(madqn.prey_number_deque_dict):
                        # ? deque? ??? ?? ??
                        last_value = value[-1]
                        wandb.log(
                            {f"Prey Count in Own Observation_{idx}": last_value, "step": iteration_number})





            if agent[:8] == "predator":

                # for each step ( doesn't change until all predators move )
                handles = env.env.env.env.env.get_handles()

                pos_predator1 = env.env.env.env.env.get_pos(handles[0])
                pos_predator2 = env.env.env.env.env.get_pos(handles[1])

                entire_pos_list = np.concatenate((pos_predator1, pos_predator2))

                # ????? ?? ?? ??
                observation, reward, termination, truncation, info = env.last()

                # predator1 ? ?? ?
                if agent[9] == "1":
                    idx = int(agent[11:])
                    pos = pos_predator1[idx]
                    view_range = predator1_view_range
                    madqn.set_agent_shared(view_range)
                    observation_temp = process_array_1(observation)
                    madqn.set_team_idx(0)

                    # dist_list = madqn.dist(observation_temp) #?? agent? ?? prey??? ??? ??? ???? dist_list ? ????.
                    # ?? ????? observation ???? prey??? ??? ?? ???, ???? ??? ??? ??? ????.

                    dist_list = np.array([np.mean(madqn.dist(observation_temp))], dtype=float)
                    print(dist_list)
                    madqn.concat_dist(dist_list)  # ?? agent? ???? ?? summation_team_dist ? prey??? ?????? ????.

                    overlap_pos_list = np.concatenate(
                        ([entire_pos_list[idx]], entire_pos_list[0:idx], entire_pos_list[(idx + 1):]))
                    overlap_tiles_pred1, overlap_tiles_pred2 = coor_list_pred1(
                        overlap_pos_list)  # ???? ????,? ?? n_predator1 -1 ?? predator1, n_predator2?? predator2


                # predator2 ? ?? ?
                else:
                    idx = int(agent[11:]) + n_predator1
                    pos = pos_predator2[idx - n_predator1]
                    view_range = predator2_view_range
                    madqn.set_agent_shared(view_range)
                    observation_temp = process_array_2(observation)
                    madqn.set_team_idx(1)

                    dist_list = np.array([np.mean(madqn.dist(observation_temp))], dtype=float)
                    madqn.concat_dist(dist_list)  # ?? agent? ???? ?? summation_team_dist ? prey??? ?????? ????.

                    # ??? ??? ??? ????,? ?? n_predator1?? predator1?? ??, n_predator2 - 1?? predator2? ??? ???? ???.
                    overlap_pos_list = np.concatenate(
                        ([entire_pos_list[idx]], entire_pos_list[0:idx], entire_pos_list[(idx + 1):]))
                    # coor_list_pred2? predator2? ??? ????? predator1? predator2? ??? ??? ??? ?? ????.
                    overlap_tiles_pred1, overlap_tiles_pred2 = coor_list_pred2(overlap_pos_list)

                madqn.set_agent_info(agent, pos, view_range)

                if termination or truncation:
                    print(agent, 'is terminated')
                    env.step(None)
                    # truncation ?? ???? iteration_number? ??? ???? ??.
                    iteration_number += 1
                    continue

                else:
                    # action : action ?
                    # book : ?? ??? ???? shared graph ?? ??? ??
                    # shared_info : ????? observation? gnn? ?? shared graph? ???? ? ???
                    # l2_before: ??? observation ? Norm ?
                    # l2_outtake : ??? Observation ? out take ??? GNN ?? ?? ??? ??????? ??? ??? ?
                    # shared_sum :  shared_info ? summation ?
                    # l2_intake :  ? ????? shared graph ?? ??? ??? ??? ????? ??? ??? ?
                    # after_gnn : shared graph ??? ??? intake ??? gnn? ?? ?? ?
                    action, book, shared_info, l2_before, l2_outtake, shared_sum, l2_intake, after_gnn = madqn.get_action(
                        state=observation_temp, mask=None)
                    env.step(action)

                    # ??? intake
                    # ???? l2_intake, l2_outtake ?? ??? ?? ???? ?????, ?? ?? ????? ???? view_range? ???? ?? ??????.
                    # ??? ?? ?????? ?? ??? ??? ??? ??? ???(outtake), ??? ??? ?????(intake) ??? ??? ??.
                    # ??? ??? ??, intake_sum ? ?? ??? ???? ??? ????.

                    # intake_sum: overpal_tiles_pred1(??? observation? predator1? observation? ?? ??? ??? ??)? ???? book ? after_gnn ? ??? ??? ??? ??? ?
                    intake_sum_with_pred1 = intake_sum(book, after_gnn,
                                                       overlap_tiles_pred1)  # ????? predator1? ??? ????? ??? intake ??? ??? ????? ?? ??? ?
                    intake_sum_with_pred2 = intake_sum(book, after_gnn,
                                                       overlap_tiles_pred2)  # ????? predator2? ??? ????? ??? intake ??? ??? ????? ?? ??? ?
                    madqn.intake_sum_with_pred1_deque_dict[idx].append(intake_sum_with_pred1)
                    madqn.intake_sum_with_pred2_deque_dict[idx].append(intake_sum_with_pred2)

                    intake_inner_with_pred1 = intake_inner(book, after_gnn,
                                                           overlap_tiles_pred1)  # ????? predator1? ??? ????? ??? intake ??? ??? ????? ?? ??? ?
                    intake_inner_with_pred2 = intake_inner(book, after_gnn,
                                                           overlap_tiles_pred2)  # ????? predator2? ??? ????? ??? intake ??? ??? ????? ?? ??? ?
                    madqn.intake_inner_with_pred1_deque_dict[idx].append(intake_inner_with_pred1)
                    madqn.intake_inner_with_pred2_deque_dict[idx].append(intake_inner_with_pred2)

                    # ?? ??? ??? ??? ??? ???? ?? ??? ??? ???, ??? ??? ?? ????? ??? ?? '??? ?'? ???? ?
                    madqn.tiles_number_with_pred1_deque_dict[idx].append(
                        len(overlap_tiles_pred1))  # ????? predator1? ??? ??? ? ??
                    madqn.tiles_number_with_pred2_deque_dict[idx].append(
                        len(overlap_tiles_pred2))  # ????? predator2? ??? ??? ? ??

                    avg_dist = madqn.avg_dist(observation_temp)  # ?? observation ?? ?? prey??? ????
                    min_dist = madqn.min_dist(observation_temp)  # ?? observation ?? ?? prey??? ????
                    madqn.agent_avg_dist_deque_dict[idx].append(avg_dist)  # ??
                    madqn.agent_min_dist_deque_dict[idx].append(min_dist)  # ??

                    number = prey_number(observation_temp)  # ?? observation ?? ?? prey ? ?
                    madqn.prey_number_deque_dict[idx].append(number)  # ??

                    madqn.shared_mean_deque_dict[idx].append(th.mean(shared_info))
                    madqn.l2_before_outtake_deque_dict[idx].append(l2_before)
                    madqn.l2_outtake_deque_dict[idx].append(l2_outtake)
                    madqn.l2_intake_deque_dict[idx].append(l2_intake)

                    # move
                    if action in [0, 1, 3, 4]:
                        move_penalty_dict[idx].append(args.move_penalty)
                        madqn.ep_move_count()  # ep ? ?? move ? ??
                        madqn.step_move_count()  # step ??  ?? ??? ???? +1 ? ???.
                        move = 1

                    # tag
                    elif action in [5, 6, 7, 8, 9, 10, 11, 12]:
                        move_penalty_dict[idx].append(0)  # ???? ???? move_penalty? 0 ??? ??.
                        madqn.step_tag_count()  # ep ? ?? tag ? ??
                        move = 2

                    # stay
                    else:
                        move_penalty_dict[idx].append(0)  # ???? ???? move_penalty? 0 ??? ??.
                        move = 0

                    madqn.agent_action_deque_dict[idx].append(move)  # ? ????? ?? action ? ??? ????.

                    reward = env._cumulative_rewards[agent]  # agent

                    observations_dict[idx].append(observation_temp)
                    action_dict[idx].append(action)
                    reward_dict[idx].append(reward)
                    book_dict[idx].append(book)
                    shared_info_dict[idx].append(shared_info)
                    termination_dict[idx].append(termination)
                    truncation_dict[idx].append(truncation)
                    agent_pos[idx].append(pos)
                    entire_pos.append(entire_pos_list)

                if madqn.buffer.size() >= args.trainstart_buffersize:
                    madqn.replay()


            else:  # prey
                observation, reward, termination, truncation, info = env.last()

                if termination or truncation:
                    print(agent, 'is terminated')
                    env.step(None)
                    iteration_number += 1
                    continue

                else:
                    action = env.action_space(agent).sample()
                    env.step(action)

            iteration_number += 1

            #  intake case1 ????? ? ??
            #  "? ????? ?? observation? ??"? "1step ??? ?? ?????? observation? ??? ??" ? ??? ???? ??
            # ?? ???? ? ???..!
            if (((iteration_number) % (args.n_predator1 + args.n_predator2 + args.n_prey)) == 0):

                # ??? ????? shared graph ? ??? ?? ??? ??? ??? 0??? ? ? ??.
                if step_idx == 1:
                    for idx in range(n_predator1 + n_predator2):
                        madqn.intake_overlap_with_pred1[idx].append(0)
                        madqn.intake_overlap_with_pred2[idx].append(0)


                # ???  ????? ??? ??? ?? ??? ??? ???? ??.
                elif step_idx != args.max_update_steps:
                    past = entire_pos[-2]  # ?? ????? ? step ??? ????
                    now = entire_pos[-1]  # ?? ????? ?? ????

                    ratio_matrix = calculate_Overlap_ratio_intake(past, now)

                    # ????? ?? observation? ?? ? predator1 ? ??? ??? ??? ?? ???? ??
                    for idx, value in enumerate(ratio_matrix[:, 0]):
                        madqn.intake_overlap_with_pred1[idx].append(value)

                    # ????? ?? observation? ?? ? predator2 ? ??? ??? ??? ?? ???? ??
                    for idx, value in enumerate(ratio_matrix[:, 1]):
                        madqn.intake_overlap_with_pred2[idx].append(value)

                # ??? ??(? truncation step)??? ??? ??? ??.
                else:
                    pass

        # ? ?????? ???? ?? wandb ??!
        if madqn.buffer.size() >= args.trainstart_buffersize:
            wandb.log({"ep_reward": ep_reward, "ep_reward_pred1": ep_reward_pred1, "ep_reward_pred2": ep_reward_pred2,
                       "ep_move_pred1": madqn.ep_move_count_pred[0], "ep_move_pred2": madqn.ep_move_count_pred[1],
                       # ?? ??? ??? ?? 0: ???? ?? 1: ??? ?? ????? ?? ? ??.
                       "(ep_reward_pred1)/(ep_move_move_pred1)": ep_reward_pred1 / madqn.ep_move_count_pred[0],
                       # ep? ???? ?? ??? reward? ???? ? ? ??.
                       "(ep_reward_pred2)/(ep_move_move_pred2)": ep_reward_pred2 / madqn.ep_move_count_pred[1]})

        if ep > args.total_ep:  # 100

            print('*' * 10, 'train over', '*' * 10)
            print(iteration_number)
            break

        if (madqn.buffer.size() >= args.trainstart_buffersize) and (ep % args.target_update == 0):
            for agent in range(args.n_predator1 + args.n_predator2):
                madqn.set_agent_model(agent)
                madqn.target_update()

        # wandb.plot
        # if (ep % args.plot_term == 0) and (ep > 0):
        #     madqn.plot(ep)

        if (ep % args.ep_save) == 0:
            for i in range(len(madqn.gdqns)):
                th.save(madqn.gdqns[i].state_dict(), 'model_save/' + 'model_' + str(i) + '_ep' + str(ep) + '.pt')
                th.save(madqn.gdqn_targets[i].state_dict(),
                        'model_save/' + 'model_target_' + str(i) + '_ep' + str(ep) + '.pt')

    print('*' * 10, 'train over', '*' * 10)


if __name__ == '__main__':
    main()

    # for i in range(len(madqn.gdqns)) :
    #     th.save(madqn.gdqns[i].state_dict(), 'model_save/'+'model_'+ str(i) +'.pt')
    #     th.save(madqn.gdqns[i].state_dict(), 'model_save/' + 'model_target_' + str(i) + '.pt')

    print('done')



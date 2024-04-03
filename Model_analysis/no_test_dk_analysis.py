from magent2.environments import hetero_adversarial_v1
from MADQN_analysis import MADQN

import numpy as np
import torch as th
import wandb

from arguments import args


device = 'cpu'

wandb.init(project="MADQN", entity='hails',config=args.__dict__)
wandb.run.name = 'analysis_1'

render_mode = 'rgb_array'
# render_mode = 'human'

predator1_view_range = args.predator1_view_range
predator2_view_range = args.predator2_view_range
n_predator1 = args.n_predator1
n_predator2 = args.n_predator2
n_prey = args.n_prey

shared_shape = (args.map_size + (predator1_view_range-2)*2, args.map_size + (predator1_view_range-2)*2, 3)
predator1_obs = (predator1_view_range*2,predator1_view_range*2, 3)
predator2_obs = (predator2_view_range*2,predator2_view_range*2, 3)
dim_act = 13

predator1_adj = ((predator1_view_range*2)**2, (predator1_view_range*2)**2)
predator2_adj = ((predator2_view_range*2)**2, (predator2_view_range*2)**2)


batch_size = 1


shared = th.zeros(shared_shape)
madqn = MADQN(n_predator1, n_predator2, predator1_obs, predator2_obs, dim_act, shared_shape, shared, args.buffer_size, device)


# for i in range(n_predator1 + n_predator2):
# 	madqn.gdqns[i].load_state_dict(th.load(f'./model_cen_save/model_{i}_ep50.pt'))

		# model_file_name = f'./model_cen_save/model_{i}_ep50.pt'
		# model_state_dict = th.load(model_file_name)
		# madqn.gdqns[i].load_state_dict(model_state_dict)


def process_array_1(arr):  #predator1 (obs, team, team_hp, predator2, predator2 hp, prey, prey hp)
    arr = np.delete(arr, [2, 4, 6], axis=2)
    combined_dim = np.logical_or(arr[:, :, 1], arr[:, :, 2])
    result = np.dstack((arr[:, :, 0], combined_dim, arr[:, :, 3]))

    return result


def process_array_2(arr): #predator2 (obs, team, team_hp, prey, prey hp, predator2, predator2 hp)
    arr = np.delete(arr, [2, 4, 6], axis=2)
    combined_dim = np.logical_or(arr[:, :, 1], arr[:, :, 3])
    result = np.dstack((arr[:, :, 0], combined_dim, arr[:, :, 2]))

    return result

def check_zero_size_min_pred1(list):

    if list.size > 0:
        # 배열이 비어 있지 않으므로 최소값 찾기
        min_value = np.min(list)
    else:
        min_value = args.predator1_view_range + 1

    return min_value

def check_zero_size_min_pred2(list):

    if list.size > 0:
        # 배열이 비어 있지 않으므로 최소값 찾기
        min_value = np.min(list)
    else:
        min_value = args.predator2_view_range + 1

    return min_value

def check_zero_size_avg_pred1(list):
    if list.size > 0:
        # 배열이 비어 있지 않으므로 최소값 찾기
        avg_value = np.mean(list)
    else:
        avg_value = args.predator1_view_range + 1

    return avg_value

def check_zero_size_avg_pred2(list):
    if list.size > 0:
        # 배열이 비어 있지 않으므로 최소값 찾기
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
        # reset ep_move_count
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


        print("ep:",ep,'*' * 80)

        for agent in env.agent_iter():

            step_idx = iteration_number // (args.n_predator1 + args.n_predator2 + args.n_prey)


            if (((iteration_number) % (args.n_predator1 + args.n_predator2 + args.n_prey)) == 0
                    and step_idx > 0):

                madqn.shared_decay()

            #book process atter all agents took actions
            # max_update_steps 이 끝나면  env.last()를 만들어서 truncation=TRUE 가 되어 해당 에이전트를 죽이는 과정이 필요하다.
            if (((iteration_number) % (args.n_predator1 + args.n_predator2 + args.n_prey)) == 0) and (iteration_number > 0):

                if step_idx == args.max_update_steps :

                    if step_idx <= args.book_term:

                        for idx in range(n_predator1 + n_predator2):

                            madqn.set_agent_pos(agent_pos[idx][-1])

                            if idx < args.n_predator1:
                                madqn.set_agent_shared(predator1_view_range)
                            else:
                                madqn.set_agent_shared(predator2_view_range)

                            # self.to_guestbook(shared_info.to('cpu'))
                            madqn.to_guestbook(shared_info_dict[idx][-1].to('cpu'))

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

                    # avg(dist(prey))-move plotting process for each team

                    # 1을 빼는 이유는 reward/move count 일때 move count = 0 이면 정의가 되지 않아서 발생하는 오류를 해결하기 위해 애초에 1을 기본값으로
                    # 지정해놨었는데, 여기서는 그런 오류가 발생한 우려가 없기 때문에 -1을 해주는 것이다.

                    # 분석을 위해 predator1 의 avg(distance)와 avg(count)데이터 버퍼에 넣기
                    madqn.avg_dist_into_deque_pred1(check_zero_size_avg_pred1(madqn.summation_team_dist[0]))
                    madqn.avg_move_into_deque_pred1((madqn.step_move_count_pred[0] - 1) / n_predator1)

                    # 분석을 위해 predator2 의 avg(distance)와 avg(count)데이터 버퍼에 넣기
                    madqn.avg_dist_into_deque_pred2(check_zero_size_avg_pred2(madqn.summation_team_dist[1]))
                    madqn.avg_move_into_deque_pred2((madqn.step_move_count_pred[1] - 1) / n_predator2)

                    # 분석을 위해 predator 의 min(distance)데이터 버퍼에 넣기
                    madqn.min_dist_into_deque_pred1(check_zero_size_min_pred1(madqn.summation_team_dist[0]))
                    madqn.min_dist_into_deque_pred2(check_zero_size_min_pred2(madqn.summation_team_dist[1]))

                    # 이렇게 버퍼에 넣어주었으므로 이제 적절한 타이밍에 plotting 을 하는 코드를 짜면 된다->밑에 있음

                    madqn.reset_step_move_count()  # 매스텝마다 이걸 지워주어야 한다.
                    madqn.reset_summation_team_dist()  # 거리도 지워주어야 한다.


                else:
                    madqn.avg_dist_into_deque_pred1(check_zero_size_avg_pred1(madqn.summation_team_dist[0]))
                    madqn.avg_move_into_deque_pred1((madqn.step_move_count_pred[0] - 1) / n_predator1)

                    # 분석을 위해 predator2 의 avg(distance)와 avg(count)데이터 버퍼에 넣기
                    madqn.avg_dist_into_deque_pred2(check_zero_size_avg_pred2(madqn.summation_team_dist[1]))
                    madqn.avg_move_into_deque_pred2((madqn.step_move_count_pred[1] - 1) / n_predator2)

                    # 분석을 위해 predator 의 min(distance)데이터 버퍼에 넣기
                    madqn.min_dist_into_deque_pred1(check_zero_size_min_pred1(madqn.summation_team_dist[0]))
                    madqn.min_dist_into_deque_pred2(check_zero_size_min_pred2(madqn.summation_team_dist[1]))

                    # 이렇게 버퍼에 넣어주었으므로 이제 적절한 타이밍에 plotting 을 하는 코드를 짜면 된다->밑에 있음

                    madqn.reset_step_move_count()  # 매스텝마다 이걸 지워주어야 한다.
                    madqn.reset_summation_team_dist()  # 거리도 지워주어야 한다.

            # put experience into the buffer after second step
            if (((iteration_number) % (args.n_predator1 + args.n_predator2 + args.n_prey)) == 0
                    and step_idx > 1 ):

                pred_step_rewards = 0
                total_move_penalty = 0

                step_reward_pred1 = 0
                step_reward_pred2 = 0


                # calculate pred_step_rewards & ep_reward
                for agent_rewards in reward_dict.values():
                    pred_step_rewards += np.sum(agent_rewards[-1])

                for penalty in move_penalty_dict.values():
                    total_move_penalty += np.sum(penalty[-2])

                pred_step_rewards = pred_step_rewards + total_move_penalty

                ep_reward += pred_step_rewards

                # sum of each team's rewards
                for i, agent_rewards in enumerate(reward_dict.values()):
                    if i < len(reward_dict) // 2:
                        step_reward_pred1 += np.sum(agent_rewards[-1])
                    else:
                        step_reward_pred2 += np.sum(agent_rewards[-1])


                ep_reward_pred1 += step_reward_pred1
                ep_reward_pred2 += step_reward_pred2



                for idx in range(n_predator1 + n_predator2):
                    # madqn.set_agent_info(agent, pos, view_range)
                    madqn.set_agent_buffer(idx)

                    madqn.buffer.put(observations_dict[idx][-2],
                                     book_dict[idx][-2],
                                     action_dict[idx][-2],
                                     pred_step_rewards,
                                     observations_dict[idx][-1],
                                     book_dict[idx][-1],
                                     termination_dict[idx][-2],
                                     truncation_dict[idx][-2])


                print('ep:{}'.format(ep))
                print("predator total_reward", pred_step_rewards)
                print("*" * 10)



                if madqn.buffer.size() >= args.trainstart_buffersize:
                    wandb.log({"pred_step_rewards": pred_step_rewards,
                               "shared_mean": madqn.shared.mean(),
                               "shared_std": madqn.shared.std()})


            if agent[:8] == "predator":

                # for each step ( doesn't change until all predators move )
                handles = env.env.env.env.env.get_handles()
                pos_predator1 = env.env.env.env.env.get_pos(handles[0])
                pos_predator2 = env.env.env.env.env.get_pos(handles[1])


                observation, reward, termination, truncation, info = env.last()



                if agent[9] == "1":
                    idx = int(agent[11:])
                    pos = pos_predator1[idx]
                    view_range = predator1_view_range
                    madqn.set_agent_shared(view_range)
                    observation_temp = process_array_1(observation)
                    madqn.set_team_idx(0)
                    dist_list = madqn.dist(observation_temp) #현재 agent와 다른 prey들과의 거리를 나타낸 리스트를 dist_list 에 저장한다.
                    madqn.put_dist(dist_list) #현재 agent의 팀의 summation_team_dist 에 리스트 정보를 concat한다. 그럼, 모든 step 이 끝났을때 팀 전체의 prey까지의 거리의 평균을 구할 수 있다.



                else:
                    idx = int(agent[11:]) + n_predator1
                    pos = pos_predator2[idx - n_predator1]
                    view_range = predator2_view_range
                    madqn.set_agent_shared(view_range)
                    observation_temp = process_array_2(observation)
                    madqn.set_team_idx(1)
                    dist_list = madqn.dist(observation_temp)
                    madqn.put_dist(dist_list)


                madqn.set_agent_info(agent, pos, view_range)
                # avg_dist = madqn.avg_dist(observation_temp) 개인에 대해서 어떻게 행동하는지 찍을 것이 아니기 때문에 당장은 필요없다.

                if termination or truncation:
                    print(agent , 'is terminated')
                    env.step(None)
                    continue

                else:
                    action, book ,shared_info = madqn.get_action(state=observation_temp, mask=None)
                    env.step(action)

                    # penalty for moving
                    if action in [0, 1, 3, 4]:
                        move_penalty_dict[idx].append(args.move_penalty)
                        madqn.ep_move_count()
                        madqn.step_move_count() #현재 agent 팀의 팀원이 움직이면 +1 을 해준다.
                        #move = 1

                    else:
                        move_penalty_dict[idx].append(0)
                        #move = 0

                    # wandb.log({"Avg_dist": avg_dist, "move": move, "title": "action distribution_per.step_{}".format(idx)})

                    reward = env._cumulative_rewards[agent] # agent


                    observations_dict[idx].append(observation_temp)
                    action_dict[idx].append(action)
                    reward_dict[idx].append(reward)
                    book_dict[idx].append(book)
                    shared_info_dict[idx].append(shared_info)
                    termination_dict[idx].append(termination)
                    truncation_dict[idx].append(truncation)
                    agent_pos[idx].append(pos)


                if madqn.buffer.size() >= args.trainstart_buffersize:
                    madqn.replay()


            else: #prey
                observation, reward, termination, truncation, info = env.last()

                if termination or truncation:
                    print(agent, 'is terminated')
                    env.step(None)
                    continue

                else:
                    action = env.action_space(agent).sample()
                    env.step(action)


            iteration_number += 1

            # if (((iteration_number) % (args.n_predator1 + args.n_predator2 + args.n_prey)) == 0
            #         and step_idx > 0 and step_idx != args.max_update_steps):
            #
            #     madqn.shared_decay()


        if madqn.buffer.size() >= args.trainstart_buffersize:
            wandb.log({"ep_reward": ep_reward,"ep_reward_pred1": ep_reward_pred1,"ep_reward_pred2": ep_reward_pred2,
                       "ep_move_pred1": madqn.ep_move_count_pred[0],"ep_move_pred2": madqn.ep_move_count_pred[1],
                       "(ep_reward_pred1)/(ep_move_move_pred1)": ep_reward_pred1/madqn.ep_move_count_pred[0],
                       "(ep_reward_pred2)/(ep_move_move_pred2)": ep_reward_pred2/madqn.ep_move_count_pred[1]})


            #사실 이것도 평균을 내서 0: 움직이지 않음 1: 움직임 으로 판단하는게 좋을 것 같다.
            # wandb.log({"ep_move_pred1": madqn.ep_move_count_pred[0]})
            # wandb.log({"ep_move_pred2": madqn.ep_move_count_pred[1]})

            # ep당 움직임에 대해 얼마나 reward를 받는지를 알 수 있다.
            # wandb.log({"(ep_reward_pred1)/(ep_move_move_pred1)": ep_reward_pred1/madqn.ep_move_count_pred[0]})
            # wandb.log({"(ep_reward_pred2)/(ep_move_move_pred2)": ep_reward_pred2/madqn.ep_move_count_pred[1]})



        if ep > args.total_ep: #100
            print('*' * 10, 'train over', '*' * 10)
            print(iteration_number)
            break



        if (madqn.buffer.size() >= args.trainstart_buffersize) and (ep % args.target_update == 0):
            for agent in range(args.n_predator1 + args.n_predator2):

                madqn.set_agent_model(agent)
                madqn.target_update()


        #wandb.plot
        if (ep % args.plot_term == 0) and (ep > 0):
            madqn.plot(ep)



        if (ep % args.ep_save) ==0 :
            for i in range(len(madqn.gdqns)) :
                th.save(madqn.gdqns[i].state_dict(), 'model_save/'+'model_'+ str(i) + '_ep' +str(ep) +'.pt')
                th.save(madqn.gdqns[i].state_dict(), 'model_save/' + 'model_target_' + str(i) + '_ep' + str(ep)+ '.pt')

    print('*' * 10, 'train over', '*' * 10)


if __name__ == '__main__':
    main()

    # for i in range(len(madqn.gdqns)) :
    #     th.save(madqn.gdqns[i].state_dict(), 'model_save/'+'model_'+ str(i) +'.pt')
    #     th.save(madqn.gdqns[i].state_dict(), 'model_save/' + 'model_target_' + str(i) + '.pt')

    print('done')



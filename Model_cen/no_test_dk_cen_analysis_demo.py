from magent2.environments import hetero_adversarial_v1
from MADQN_analysis import MADQN
from scipy.spatial.distance import pdist, squareform
import numpy as np
import torch as th
import wandb

from arguments import args


device = 'cpu'

# wandb.init(project="MADQN", entity='hails',config=args.__dict__)
# wandb.run.name = 'wandb test'

render_mode = 'rgb_array'
#render_mode = 'human'
#env = hetero_adversarial_v1.env(map_size=args.map_size, minimap_mode=False, tag_penalty=-0.2,
# max_cycles=args.max_update_steps, extra_features=False,render_mode=render_mode)

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

# predator1_adj = (predator1_view_range*2,predator1_view_range*2)
# predator2_adj = (predator2_view_range*2,predator2_view_range*2)


batch_size = 1

target_update_point = (1+args.max_update_steps)*(args.n_predator1+args.n_predator2+args.n_prey)


shared = th.zeros(shared_shape)
madqn = MADQN(n_predator1, n_predator2, predator1_obs, predator2_obs, dim_act, shared_shape, shared, args.buffer_size, device)


def process_array_1(arr):  #predator1 (obs, ??, ??hp, predator2, predator2 hp, prey, prey hp)
    arr = np.delete(arr, [2, 4, 6], axis=2)
    combined_dim = np.logical_or(arr[:, :, 1], arr[:, :, 2])
    result = np.dstack((arr[:, :, 0], combined_dim, arr[:, :, 3]))

    return result


def process_array_2(arr): #predator2 (obs, ??, ??hp, prey, prey hp, predator2, predator2 hp)
    arr = np.delete(arr, [2, 4, 6], axis=2)
    combined_dim = np.logical_or(arr[:, :, 1], arr[:, :, 3])
    result = np.dstack((arr[:, :, 0], combined_dim, arr[:, :, 2]))

    return result


def main():

    for ep in range(args.total_ep):

        ep_reward = 0
        ep_reward_pred1 = 0
        ep_reward_pred2 = 0
        iteration_number = 0

        # for
        madqn.ep_move_count_reset()


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


        env = hetero_adversarial_v1.env(map_size=args.map_size, minimap_mode=False, tag_penalty=args.tag_penalty,
                                        max_cycles=args.max_update_steps, extra_features=False, render_mode=render_mode)

        # ep ??? ???? shared ????.
        madqn.reset_shred(shared)
        env.reset(seed=args.seed)


        print("ep:",ep,'*' * 80)

        for agent in env.agent_iter():

            step_idx = iteration_number // (args.n_predator1 + args.n_predator2 + args.n_prey)

            # # for each step ( position doesn't change until all predators move )
            # handles = env.env.env.env.env.get_handles()
            # pos_predator1 = env.env.env.env.env.get_pos(handles[0])
            # pos_predator2 = env.env.env.env.env.get_pos(handles[1])
            # pos_prey = env.env.env.env.env.get_pos(handles[2])
            # pos_entire_list = np.vstack((pos_predator1, pos_predator2, pos_prey))
            #
            # distance_matrix = squareform(pdist(pos_entire_list, 'euclidean'))
            # dist_between_pred1_prey = distance_matrix[:n_predator1, (n_predator1 + n_predator2):]
            # dist_between_pred2_prey = distance_matrix[n_predator1:(n_predator1 + n_predator2),
            #                           (n_predator1 + n_predator2):]

            #book process & avg(dist(prey))-move plotting process for each team
            if (((iteration_number) % (args.n_predator1 + args.n_predator2 + args.n_prey)) == 0) and (iteration_number > 0):

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

                        madqn.set_agent_pos(agent_pos[idx][-(args.book_term+1)])

                        if idx < args.n_predator1:
                            madqn.set_agent_shared(predator1_view_range)
                        else:
                            madqn.set_agent_shared(predator2_view_range)

                        # self.to_guestbook(shared_info.to('cpu'))
                        madqn.to_guestbook(-(args.book_decay**(args.book_term))*shared_info_dict[idx][-(args.book_term+1)].to('cpu'))

                    # Add recent Step information
                    for idx in range(n_predator1 + n_predator2):

                        madqn.set_agent_pos(agent_pos[idx][-1])

                        if idx < args.n_predator1:
                            madqn.set_agent_shared(predator1_view_range)
                        else:
                            madqn.set_agent_shared(predator2_view_range)

                        madqn.to_guestbook(shared_info_dict[idx][-1].to('cpu'))

                # 이 위치에 avg(dist(prey))-move plotting process for each team 있어야함
                #### predator observation 안에 있는 prey에 대해서만 구했어야 했는데 그냥 전체 observation에 대해서만 구해서 잘못됐다....
                wandb.log({"Avg_dist": np.mean(dist_between_pred1_prey), "move": np.mean(madqn.step_move_count_pred[0]),
                           "title": "pred1 action distribution per step"})
                wandb.log({"Avg_dist": np.mean(dist_between_pred2_prey), "move": np.mean(madqn.step_move_count_pred[1]),
                           "title": "pred2 action distribution per step"})

            # put experience into the buffer after second step
            if ((((iteration_number) % (args.n_predator1 + args.n_predator2 + args.n_prey)) == 0)
                    and step_idx > 1):

                total_last_rewards = 0
                total_move_penalty = 0

                step_reward_pred1 = 0
                step_reward_pred2 = 0


                # calculate total_last_rewards & ep_reward
                for agent_rewards in reward_dict.values():
                    total_last_rewards += np.sum(agent_rewards[-1])

                for penalty in move_penalty_dict.values():
                    total_move_penalty += np.sum(penalty[-2])

                total_last_rewards = total_last_rewards + total_move_penalty

                ep_reward += total_last_rewards

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
                                     total_last_rewards,
                                     observations_dict[idx][-1],
                                     book_dict[idx][-1],
                                     termination_dict[idx][-2],
                                     truncation_dict[idx][-2])





                print('ep:{}'.format(ep))
                print("predator total_reward", total_last_rewards)
                print("*" * 10)



                if madqn.buffer.size() >= args.trainstart_buffersize:
                    wandb.log({"total_last_rewards": total_last_rewards})
                    wandb.log({"shared_mean": madqn.shared.mean()})
                    wandb.log({"shared_std": madqn.shared.std()})





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
                    observation_temp = process_array_1(observation)
                    madqn.set_team_idx(0)
                else:
                    idx = int(agent[11:]) + n_predator1
                    pos = pos_predator2[idx - n_predator1]
                    view_range = predator2_view_range
                    observation_temp = process_array_2(observation)
                    madqn.set_team_idx(1)

                madqn.set_agent_info(agent, pos, view_range)


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
                        madqn.step_move_count()
                        move = 1

                    else:
                        move_penalty_dict[idx].append(0)
                        move = 0

                    #각 predator 에 대해서 prey의 위치에 대해서 move를 하는지 하지 않는지에 대해서 구한건데, 이것역시 전체 observation 에 대해서 avg(prey distance)를 구한거라서 잘못됐다.
                    wandb.log({"Avg_dist": np.mean(distance_matrix[idx,(n_predator1+n_predator2):]), "move": move, "title": "action distribution_per.step_{}".format(idx)})

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

            if ((((iteration_number) % (args.n_predator1 + args.n_predator2 + args.n_prey)) == 0)
                    and step_idx > 0):

                madqn.shared_decay()


        if madqn.buffer.size() >= args.trainstart_buffersize:
            wandb.log({"ep_reward": ep_reward})
            wandb.log({"ep_reward_pred1": ep_reward_pred1})
            wandb.log({"ep_reward_pred2": ep_reward_pred2})
            wandb.log({"ep_move_move_pred1": madqn.ep_move_count_pred[0]})
            wandb.log({"ep_move_move_pred2": madqn.ep_move_count_pred[1]})


            wandb.log({"(ep_reward_pred1)/(ep_move_move_pred1)": ep_reward_pred1/madqn.ep_move_count_pred[0]})
            wandb.log({"(ep_reward_pred2)/(ep_move_move_pred2)": ep_reward_pred2/madqn.ep_move_count_pred[1]})



        if ep > args.total_ep: #100
            print('*' * 10, 'train over', '*' * 10)
            print(iteration_number)
            break



        if (madqn.buffer.size() >= args.trainstart_buffersize) and (ep % args.target_update == 0):
            for agent in range(args.n_predator1 + args.n_predator2):

                madqn.set_agent_model(agent)
                madqn.target_update()


        # if (ep % args.ep_save) ==0 :
        #     for i in range(len(madqn.gdqns)) :
        #         th.save(madqn.gdqns[i].state_dict(), 'model_save/'+'model_'+ str(i) + '_ep' +str(ep) +'.pt')
        #         th.save(madqn.gdqns[i].state_dict(), 'model_save/' + 'model_target_' + str(i) + '_ep' + str(ep)+ '.pt')

    print('*' * 10, 'train over', '*' * 10)


if __name__ == '__main__':
    main()

    # for i in range(len(madqn.gdqns)) :
    #     th.save(madqn.gdqns[i].state_dict(), 'model_save/'+'model_'+ str(i) +'.pt')
    #     th.save(madqn.gdqns[i].state_dict(), 'model_save/' + 'model_target_' + str(i) + '.pt')

    print('done')



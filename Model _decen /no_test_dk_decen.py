from magent2.environments import hetero_adversarial_v1
from MADQN_decen import MADQN
import argparse
import numpy as np
import torch as th
import wandb

from arguments import args


device = 'cpu'

# wandb.init(project="MADQN", entity='hails',config=args.__dict__)
# wandb.run.name = 'revision_gnn_8'

render_mode = 'rgb_array'
#render_mode = 'human'
#env = hetero_adversarial_v1.env(map_size=args.map_size, minimap_mode=False, tag_penalty=-0.2,
# max_cycles=args.max_update_steps, extra_features=False,render_mode=render_mode)

predator1_view_range = args.predator1_view_range
predator2_view_range = args.predator2_view_range
n_predator1 = args.n_predator1
n_predator2 = args.n_predator2
n_prey = args.n_prey


predator1_obs = (predator1_view_range*2,predator1_view_range*2, 3)
predator2_obs = (predator2_view_range*2,predator2_view_range*2, 3)
dim_act = 13

predator1_adj = ((predator1_view_range*2)**2, (predator1_view_range*2)**2)
predator2_adj = ((predator2_view_range*2)**2, (predator2_view_range*2)**2)

batch_size = 1

target_update_point = (1+args.max_update_steps)*(args.n_predator1+args.n_predator2+args.n_prey)



madqn = MADQN(n_predator1, n_predator2, predator1_obs, predator2_obs, dim_act, args.buffer_size, device)


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
        iteration_number = 0

        observations_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            observations_dict[agent_idx] = []

        reward_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            reward_dict[agent_idx] = []

        action_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            action_dict[agent_idx] = []

        termination_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            termination_dict[agent_idx] = []

        truncation_dict = {}
        for agent_idx in range(n_predator1 + n_predator2):
            truncation_dict[agent_idx] = []


        env = hetero_adversarial_v1.env(map_size=args.map_size, minimap_mode=False, tag_penalty=args.tag_penalty,
                                        max_cycles=args.max_update_steps, extra_features=False, render_mode=render_mode)
        env.reset(seed=args.seed)


        print("ep:",ep,'*' * 80)

        for agent in env.agent_iter():

            step_idx = iteration_number // (args.n_predator1 + args.n_predator2 + args.n_prey)

            if agent[:8] == "predator":

                # handles = env.env.env.env.env.get_handles()
                # pos_predator1 = env.env.env.env.env.get_pos(handles[0])
                # pos_predator2 = env.env.env.env.env.get_pos(handles[1])


                observation, reward, termination, truncation, info = env.last()



                if agent[9] == "1":
                    idx = int(agent[11:])
                    observation_temp = process_array_1(observation)
                else:
                    idx = int(agent[11:]) + n_predator1
                    observation_temp = process_array_2(observation)

                madqn.set_agent_info(agent)


                if termination or truncation:
                    print(agent , 'is terminated')
                    env.step(None)
                    continue

                else:
                    action = madqn.get_action(state=observation_temp, mask=None)

                    env.step(action)
                    reward = env._cumulative_rewards[agent] # agent


                    observations_dict[idx].append(observation_temp)
                    action_dict[idx].append(action)
                    reward_dict[idx].append(reward)
                    termination_dict[idx].append(termination)
                    truncation_dict[idx].append(truncation)



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


            if ((((iteration_number + 1) % (args.n_predator1 + args.n_predator2 + args.n_prey)) == 0)
                    and step_idx > 0):

                total_last_rewards = 0
                for agent_rewards in reward_dict.values():
                    total_last_rewards += np.sum(agent_rewards[-1])

                ep_reward += total_last_rewards


                for idx in range(n_predator1 + n_predator2):
                    # madqn.set_agent_info(agent, pos, view_range)
                    madqn.set_agent_buffer(idx)

                    madqn.buffer.put(observations_dict[idx][-2],
                                     action_dict[idx][-2],
                                     total_last_rewards,
                                     observations_dict[idx][-1],
                                     termination_dict[idx][-2],
                                     truncation_dict[idx][-2])

                    # wandb.log({"action_{}".format(idx): action_dict[idx][-2]})



                print('ep:{}'.format(ep))
                print("predator total_reward", total_last_rewards)
                print("*" * 10)



                # if madqn.buffer.size() >= args.trainstart_buffersize:
                #     wandb.log({"total_last_rewards": total_last_rewards})
                #     wandb.log({"shared_mean": madqn.shared.mean()})
                #     wandb.log({"shared_std": madqn.shared.std()})

            iteration_number += 1

        # if madqn.buffer.size() >= args.trainstart_buffersize:
        #     wandb.log({"ep_reward": ep_reward})


        if ep > args.total_ep: #100
            print('*' * 10, 'train over', '*' * 10)
            print(iteration_number)
            break


        if (madqn.buffer.size() >= args.trainstart_buffersize) and (ep % args.target_update == 0):
            for agent in range(args.n_predator1 + args.n_predator2):
                madqn.set_agent_buffer(agent)
                madqn.target_update()


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



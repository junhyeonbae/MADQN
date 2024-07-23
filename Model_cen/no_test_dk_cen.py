from magent2.environments import hetero_adversarial_v1
from MADQN_cen import MADQN
from arguments import args
import torch as th
import wandb
import tqdm
import numpy as np

seed = 2000

device = 'cpu'

# wandb.init(project="MADQN", entity='hails',config=args.__dict__)
# wandb.run.name = 'cen_move_penalty_6'


render_mode = 'rgb_array'
# render_mode = "human"

entire_state = (args.map_size,args.map_size,args.dim_feature)
dim_act = 13
n_predator1 = args.n_predator1
n_predator2 = args.n_predator1
n_prey = args.n_prey



madqn = MADQN(n_predator1, n_predator2, dim_act ,entire_state, device, buffer_size=args.buffer_size)

# for i in range(n_predator1 + n_predator2):
# 	madqn.gdqns[i].load_state_dict(th.load(f'./model_cen_save/model_{i}_ep50.pt'))

		# model_file_name = f'./model_cen_save/model_{i}_ep50.pt'
		#
		# # ?? ?? ?? ??
		# model_state_dict = th.load(model_file_name)
		#
		# # ?? ?? ??? ?? ??? ??
		# madqn.gdqns[i].load_state_dict(model_state_dict)



def process_array(arr):  #predator1 (obs, team, team_hp, predator2, predator2 hp, prey, prey hp)

	arr = np.delete(arr, [2, 4, 6], axis=2)
	result = np.dstack((arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]))

	pos_list1 = np.argwhere(result[:, :, 1] == 1)
	pos_list2 = np.argwhere(result[:, :, 2] == 1)

	for i in pos_list1:
		result[i[0] + 1, i[1], 1] = 1
		result[i[0], i[1] + 1, 1] = 1
		result[i[0] + 1, i[1] + 1, 1] = 1

	for i in pos_list2:
		result[i[0] + 1, i[1], 2] = 1
		result[i[0], i[1] + 1, 2] = 1
		result[i[0] + 1, i[1] + 1, 2] = 1

	return result

def main():

	for ep in range(args.total_ep):

		iteration_number = 0  #ep? ?? ?????? 0 ?? ???
		ep_reward = 0

		env = hetero_adversarial_v1.env(map_size=args.map_size, minimap_mode=False, tag_penalty=args.tag_penalty,
										max_cycles=args.max_update_steps, extra_features=False, render_mode=render_mode)

		env.reset(seed=args.seed)
		print("ep:",ep,'*' * 80)


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


		# 'max_update_steps' ??? ??? ?? ?????.
		for agent in env.agent_iter():

			step_idx = iteration_number // (args.n_predator1 + args.n_predator2 + args.n_prey)
			# print(agent[:8]=='predator')

			if (((iteration_number) % (args.n_predator1 + args.n_predator2 + args.n_prey)) == 0
					and step_idx > 1 and step_idx != args.max_update_steps):

				total_last_rewards = 0
				total_move_penalty = 0
				for agent_rewards in reward_dict.values():
					total_last_rewards += np.sum(agent_rewards[-1])

				for penalty in move_penalty_dict.values():
					total_move_penalty += np.sum(penalty[-2])

				total_last_rewards = total_last_rewards + total_move_penalty

				ep_reward += total_last_rewards


				#??? put
				for idx in range(args.n_predator1 + args.n_predator2):

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
				print("*"*10)

				# if madqn.buffer.size() >= args.trainstart_buffersize:
				# 	wandb.log({"total_last_rewards": total_last_rewards })


			_, reward, termination, truncation, info = env.last()
			observation = env.state()
			observation_temp = process_array(observation)

			if agent[:8] == "predator":

				if agent[9] == "1":
					idx = int(agent[11:])

				else:
					idx = int(agent[11:]) + n_predator1

				madqn.set_agent_info(agent)

				if termination or truncation:
					print(agent , 'is terminated')
					env.step(None)
					continue

				else:
					action = madqn.get_action(state=observation_temp, mask=None)
					env.step(action)
					reward = env._cumulative_rewards[agent] # agent

					# ??? 0,1,2,3,4(????) ? ??? reward ? ??? ???? ??.
					if action in [0, 1, 3, 4]:
						move_penalty_dict[idx].append(args.move_penalty)
					else:
						move_penalty_dict[idx].append(0)


					observations_dict[idx].append(observation_temp) #s
					action_dict[idx].append(action)					#a
					reward_dict[idx].append(reward)					#r
					termination_dict[idx].append(termination)		#t_{t-1]
					truncation_dict[idx].append(truncation)			#t_{t-1]

					if madqn.buffer.size() >= args.trainstart_buffersize:

						madqn.replay()

			else : #prey ??
				_, _, termination, truncation, _ = env.last()

				if termination or truncation:
					print(agent, 'is terminated')
					env.step(None)

					continue


				else:
					action = env.action_space(agent).sample()
					env.step(action)


			iteration_number += 1

		# if madqn.buffer.size() >= args.trainstart_buffersize:
		# 	wandb.log({"ep_reward": ep_reward})



		#ep_reward += total_last_rewards
		#print("ep_reward:", ep_reward)

		# if iteration_number > args.max_update_steps:
		# 	print('*' * 10, 'train over', '*' * 10)
		# 	print(iteration_number)
		# 	break


		if ep > args.total_ep: #100
			print('*' * 10, 'train over', '*' * 10)
			print(iteration_number)

			break

		# if madqn.buffer.size() >= args.trainstart_buffersize:
		# ep? 100? ??? ??? target update ??.
		if (madqn.buffer.size() >= args.trainstart_buffersize) and (ep % args.target_update == 0):
			for agent in range(args.n_predator1 + args.n_predator2):
				madqn.set_agent_model(agent)
				madqn.target_update()

		# if ((ep % 50) ==0) and ep >1 :
		# 	for i in range(len(madqn.gdqns)) :
		# 		th.save(madqn.gdqns[i].state_dict(), './model_cen_save/'+'model_'+ str(i) + '_ep' +str(ep) +'.pt')
		# 		th.save(madqn.gdqn_targets[i].state_dict(), './model_cen_save/' + 'model_target_' + str(i) + '_ep' + str(ep)+ '.pt')

	print('*' * 10, 'train over', '*' * 10)

if __name__ == '__main__':
	main()
	#print('done')
	#??? ??# for i in range(len(madqn.gdqns)) :
	# 	th.save(madqn.gdqns[i].state_dict(), 'model_save/'+'model_'+ str(i) +'.pt')
	# 	th.save(madqn.gdqns[i].state_dict(), 'model_save/' + 'model_target_' + str(i) + '.pt')


	print('done')

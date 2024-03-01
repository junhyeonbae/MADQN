from magent2.environments import hetero_adversarial_v1
from MADQN_decen import MADQN
from arguments import args
import torch as th
import wandb
import tqdm
import numpy as np

seed = 2000

device = 'cpu'

# wandb.init(project="MADQN", entity='hails',config=args.__dict__)
# wandb.run.name = 'mac_8'


render_mode = 'rgb_array'
#render_mode = "human"

entire_state = (args.map_size,args.map_size,3)
dim_act = 13
n_predator1 = args.n_predator1
n_predator2 = args.n_predator1
n_prey = args.n_prey

#target_update_point 필요한 이유?
#iteration_number는 하나의 에이전트에 해당하는 for문이 돌아갈 때 +1 되는데,
#target 을 max_update_steps 에 맞춰서 업데이트 하기 위해서 필요하다.
target_update_point = (1+args.max_update_steps)*(args.n_predator1+args.n_predator2+args.n_prey)


madqn = MADQN(n_predator1, n_predator2, dim_act ,entire_state, device, buffer_size=args.buffer_size)
# for i in range(n_predator1 + n_predator2):
# 	madqn.gdqns[i].load_state_dict(th.load(f'./model_cen_save/model_{i}_ep50.pt'))

		# model_file_name = f'./model_cen_save/model_{i}_ep50.pt'
		#
		# # 모델 상태 사전 로드
		# model_state_dict = th.load(model_file_name)
		#
		# # 모델 상태 사전을 현재 모델에 적용
		# madqn.gdqns[i].load_state_dict(model_state_dict)

def process_array(arr):

	arr = np.delete(arr, [2, 4, 6], axis=2)
	combined_dim = np.logical_or(arr[:, :, 1], arr[:, :, 2])
	result = np.dstack((arr[:, :, 0], combined_dim, arr[:, :, 3]))

	pos_list = np.argwhere(result[:, :, 1] == 1)

	for i in pos_list:

		result[i[0]+1, i[1], 1] = 1
		result[i[0], i[1]+1, 1] = 1
		result[i[0]+1, i[1]+1, 1] = 1


	return result


# def process_array_2(arr):
#
# 	arr = np.delete(arr, [0, 2, 4, 6], axis=2)
# 	combined_dim = np.logical_or(arr[:, :, 1], arr[:, :, 2])
# 	result = np.dstack((arr[:, :, 0], combined_dim))
#
# 	return result



def main():

	for ep in range(args.total_ep):

		iteration_number = 0  #ep가 새로 시작될때마다 0 으로 초기화
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

		action_dict = {}
		for agent_idx in range(n_predator1 + n_predator2):
			action_dict[agent_idx] = []

		termination_dict = {}
		for agent_idx in range(n_predator1 + n_predator2):
			termination_dict[agent_idx] = []

		truncation_dict = {}
		for agent_idx in range(n_predator1 + n_predator2):
			truncation_dict[agent_idx] = []


		# 'max_update_steps' 끝나면 나오게 되는 반목문이다.
		for agent in env.agent_iter():

			step_idx = iteration_number // (args.n_predator1 + args.n_predator2 + args.n_prey)
			# print(agent[:8]=='predator')

			_, reward, termination, truncation, info = env.last()
			observation = env.state()
			observation_temp = process_array(observation)

			if agent[:8] == "predator":

				if agent[9] == "1":

					idx = int(agent[11:])


				else:
					idx = int(agent[11:]) + n_predator1


				madqn.set_agent_info(agent)




				# print('BEFORE: ', agent, 's reward', reward)
				# print("agent:", agent)
				# if reward > 0:
				# 	print('before reward:', reward)
				# elif reward < 0:
				# 	print('before penalty:', reward)

				if termination or truncation:
					print(agent , 'is terminated')
					env.step(None)
					continue

				else:
					action = madqn.get_action(state=observation_temp, mask=None)
					env.step(action)
					reward = env._cumulative_rewards[agent] # agent
					# ss , reward, terter, trutru, ifif = env.last()
					print('AFTER: ', agent, 'action',action, 's reward', reward)

					# if rr > 0:
					# 	print('after reward:', rr)
					# elif rr < 0:
					# 	print('after penalty:', rr)

					if reward <= -0.5:
						print('wtf')

					observations_dict[idx].append(observation_temp) #s
					action_dict[idx].append(action)					#a
					reward_dict[idx].append(reward)					#r
					termination_dict[idx].append(termination)		#t_{t-1]
					truncation_dict[idx].append(truncation)			#t_{t-1]

					if madqn.buffer.size() >= args.trainstart_buffersize:

						madqn.replay()

			else : #prey 일때
				_, _, termination, truncation, _ = env.last()

				if termination or truncation:
					print(agent, 'is terminated')
					env.step(None)

					continue


				else:
					action = env.action_space(agent).sample()
					env.step(action)

			# 이렇게 하는 이유는 팀 전체의 reward를 put 하기 위해서인데..
			# 한 step 이 끝날때마다 첫번째 agent 에 대해서 딱 한번!
			# 굳이 이렇게 if문을 또 돌릴 필요가 있나?

			if ((((iteration_number + 1) % (args.n_predator1 + args.n_predator2 + args.n_prey)) == 0)
					and step_idx > 1):  # 세번째 step 이후, 각 에이전트의 iteraiton이 끝난 직후

				total_last_rewards = 0
				for agent_rewards in reward_dict.values():
					total_last_rewards += np.sum(agent_rewards[-1])
					# last_reward = agent_rewards[-2] - agent_rewards[-3]
					# total_last_rewards += last_reward
					#
					# if last_reward <= -0.5:
					# 	print('wtf')

				if total_last_rewards <= -0.5:
					print('wtf')

				ep_reward += total_last_rewards


				#이렇게 put 하는게 맞나?
				for idx in range(args.n_predator1 + args.n_predator2):

					# madqn.set_agent_info(agent, pos, view_range)

					madqn.set_agent_buffer(idx)

					madqn.buffer.put(observations_dict[idx][-2],
									 action_dict[idx][-2],
									 total_last_rewards,
									 observations_dict[idx][-1],
									 termination_dict[idx][-2],
									 truncation_dict[idx][-2])


				print('ep:{}'.format(ep))
				print("predator total_reward", total_last_rewards)
				print("*"*10)

				# if madqn.buffer.size() >= args.trainstart_buffersize:
				# 	wandb.log({"total_last_rewards": total_last_rewards })


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
		# ep가 100의 개수일 때마다 target update 한다.
		if madqn.buffer.size() >= args.trainstart_buffersize:
			for agent in range(args.n_predator1 + args.n_predator2):
				madqn.set_agent_buffer(agent)
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

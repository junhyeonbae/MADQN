from magent2.environments import hetero_adversarial_v1
from arguments import args

render_mode = 'human'
# render_mode = 'rgb_array'
env = hetero_adversarial_v1.env(map_size=args.map_size, minimap_mode=False, tag_penalty=args.tag_penalty,
                                    max_cycles=args.max_update_steps, extra_features=False, render_mode=render_mode)

n_predator1 = args.n_predator1
n_predator2 = args.n_predator2
n_prey = args.n_prey

for ep in range(1):

	observations_dict = {}
	for agent_idx in range(n_predator1 + n_predator2 + n_prey):
		observations_dict[agent_idx] = []

	reward_dict = {}
	for agent_idx in range(n_predator1 + n_predator2+ n_prey):
		reward_dict[agent_idx] = []

	action_dict = {}
	for agent_idx in range(n_predator1 + n_predator2+ n_prey):
		action_dict[agent_idx] = []

	termination_dict = {}
	for agent_idx in range(n_predator1 + n_predator2+ n_prey):
		termination_dict[agent_idx] = []

	truncation_dict = {}
	for agent_idx in range(n_predator1 + n_predator2+ n_prey):
		truncation_dict[agent_idx] = []

	env.reset()

	iteration_number = 0

	for agent in env.agent_iter():

		step_idx = iteration_number // (args.n_predator1 + args.n_predator2 + args.n_prey)
		#print(step_idx)
		if agent[:8] == "predator":
			print("agent:", agent[11])
			print(agent)

		else:
			print(agent)
			print("agent:", agent[11])
			print(agent)


		s = env.state()
		observation, reward, termination, truncation, info = env.last() # 가장 최근의 상태를 업데이트 해주는 것

		if termination or truncation:
			print(agent, ' is terminated')
			env.step(None) # need this
			#continue #아예 현재 루프를 건더뛰게 된다.
			iteration_number += 1
		else:
			action = env.action_space(agent).sample() #위에서 받은 observation 에 따라서 action 을 선택하고 step 을 취해주면 된다.
			env.step(action)
			iteration_number += 1

		if agent[:8] == "predator":

			if agent[9] == "1":
				agent_idx = int(agent[11:])
			else:
				agent_idx = int(agent[11:]) + n_predator1
		else:
			agent_idx = int(agent[7:]) + n_predator1 + n_predator2

		print(agent_idx)

		observations_dict[agent_idx].append(observation)
		reward_dict[agent_idx].append(reward)
		action_dict[agent_idx].append(action)
		termination_dict[agent_idx].append(termination)
		truncation_dict[agent_idx].append(truncation)

	print("last")




# env.state() # receives the entire state
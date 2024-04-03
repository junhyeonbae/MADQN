from magent2.environments import hetero_adversarial_v1
from arguments import args

# render_mode = 'human'
render_mode = 'rgb_array'
env = hetero_adversarial_v1.env(map_size=args.map_size, minimap_mode=False, tag_penalty=args.tag_penalty,
                                    max_cycles=args.max_update_steps, extra_features=False, render_mode=render_mode)

for ep in range(1000):

	env.reset()
	iteration_number = 0

	for agent in env.agent_iter():

		step_idx = iteration_number // (args.n_predator1 + args.n_predator2 + args.n_prey)

		if (((iteration_number) % (args.n_predator1 + args.n_predator2 + args.n_prey)) == 0) and (iteration_number > 0)\
				and ( step_idx == args.max_update_steps):

			print("step_idx:{},iteration_number:{},ep:{},agent:{}".format(step_idx,iteration_number,ep,agent))


		observation, reward, termination, truncation, info = env.last()

		if termination or truncation:
			print(agent, ' is terminated')
			env.step(None) # need this
			iteration_number += 1
			continue
		else:
			action = env.action_space(agent).sample()
			env.step(action)

		iteration_number += 1

	# env.state() # receives the entire state


	#max_update_steps 이 끝나면  env.last()를 만들어서 truncation=TRUE 가 되어 해당 에이전트를 죽이는 과정이 필요하다.
# noqa
"""
# Adversarial Pursuit

```{figure} magent_adversarial_pursuit.gif
:width: 140px
:name: adversarial_pursuit
```

This environment is part of the <a href='..'>MAgent2 environments</a>. Please read that page first for general information.

| Import             | `from magent2.environments import adversarial_pursuit_v4` |
|--------------------|--------------------------------------------------------|
| Actions            | Discrete                                               |
| Parallel API       | Yes                                                    |
| Manual Control     | No                                                     |
| Agents             | `agents= [predator_[0-24], prey_[0-49]]`               |
| Agents             | 75                                                     |
| Action Shape       | (9),(13)                                               |
| Action Values      | Discrete(9),(13)                                       |
| Observation Shape  | (9,9,5), (10,10,9)                                     |
| Observation Values | [0,2]                                                  |
| State Shape        | (45, 45, 5)                                            |
| State Values       | (0, 2)                                                 |

```{figure} ../../_static/img/aec/magent_adversarial_pursuit_aec.svg
:width: 200px
:name: adversarial_pursuit
```

The red agents must navigate the obstacles and tag (similar to attacking, but without damaging) the blue agents. The blue agents should try to avoid being tagged. To be effective, the red agents, who are much are slower and larger than the blue agents, must work together to trap blue agents so
they can be tagged continually.

### Arguments

``` python
adversarial_pursuit_v4.env(map_size=45, minimap_mode=False, tag_penalty=-0.2,
max_cycles=500, extra_features=False)
```

`map_size`: Sets dimensions of the (square) map. Increasing the size increases the number of agents. Minimum size is 7.

`minimap_mode`: Turns on global minimap observations. These observations include your and your opponents piece densities binned over the 2d grid of the observation space. Also includes your `agent_position`, the absolute position on the map (rescaled from 0 to 1).

`tag_penalty`:  reward when red agents tag anything

`max_cycles`:  number of frames (a step for each agent) until game terminates

`extra_features`: Adds additional features to observation (see table). Default False

#### Action Space

Key: `move_N` means N separate actions, one to move to each of the N nearest squares on the grid.

Predator action options: `[do_nothing, move_4, tag_8]`

Prey action options: `[do_nothing, move_8]`

#### Reward

Predator's reward is given as:

* 1 reward for tagging a prey
* -0.2 reward for tagging anywhere (`tag_penalty` option)

Prey's reward is given as:

* -1 reward for being tagged


#### Observation space

The observation space is a 10x10 map for pursuers and a 9x9 map for the pursued. They contain the following channels, which are (in order):

feature | number of channels
--- | ---
obstacle/off the map| 1
my_team_presence| 1
my_team_hp| 1
other_team_presence| 1
other_team_hp| 1
binary_agent_id(extra_features=True)| 10
one_hot_action(extra_features=True)| 9/Prey,13/Predator
last_reward(extra_features=True)| 1

### State space

The observation space is a 45x45 map. It contains the following channels, which are (in order):

feature | number of channels
--- | ---
obstacle map| 1
prey_presence| 1
prey_hp| 1
predator_presence| 1
predator_hp| 1
binary_agent_id(extra_features=True)| 10
one_hot_action(extra_features=True)|  13 (max action space)
last_reward(extra_features=True)| 1


### Version History

* v4: Underlying library fix (1.18.0)
* v3: Fixed bugs and changed default parameters (1.7.0)
* v2: Observation space bound fix, bumped version of all environments due to adoption of new agent iteration scheme where all agents are iterated over after they are done (1.4.0)
* v1: Agent order under death changed (1.3.0)
* v0: Initial versions release (1.0.0)

"""

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_to_aec_wrapper

from arguments import args

import magent2
from magent2.environments.magent_env import magent_parallel_env, make_env

default_map_size = 45
max_cycles_default = 500
minimap_mode_default = False
default_reward_args = dict(tag_penalty=-0.2)

# env = hetero_adversarial_v1.env(map_size=45, minimap_mode=False, tag_penalty=-0.2,
# max_cycles=500, extra_features=False,render_mode=render_mode)
# 이 인자들이 raw_env에 가게 된다.
# tag_penalty=-0.2 와 render_mode=render_mode가 reward_args에 들어가 딕셔너리의 형태로 저장된다.
# 결국 tag_penalty=-0.2 만 reward_args에 딕셔너리 형태로 저장된다.

def parallel_env(
    map_size=default_map_size,
    max_cycles=max_cycles_default,
    minimap_mode=minimap_mode_default,
    extra_features=False,
    render_mode=None,
    **reward_args
):
    env_reward_args = dict(**default_reward_args)
    env_reward_args.update(reward_args)
    return _parallel_env(
        map_size, minimap_mode, env_reward_args, max_cycles, extra_features, render_mode
    )

def raw_env(
    map_size=default_map_size,
    max_cycles=max_cycles_default,
    minimap_mode=minimap_mode_default,
    extra_features=False,
    **reward_args
):
    return parallel_to_aec_wrapper(
        parallel_env(map_size, max_cycles, minimap_mode, extra_features, **reward_args)
    )


env = make_env(raw_env)
def get_config(map_size, minimap_mode, tag_penalty):
    gw = magent2.gridworld
    cfg = gw.Config()


                                    #cfg.set( )에 넘겨준 인자들(딕셔너리들)을 config_dict에 그대로 복사한다.
    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": minimap_mode})
    cfg.set({"embedding_size": 10})


    options1 = {
        "width": 2,
        "length": 2,
        "hp": 1,
        "speed": 1,
        "view_range": gw.CircleRange(args.predator1_view_range),
        "attack_range": gw.CircleRange(2),
        "attack_penalty": tag_penalty,
    }

    predator = [cfg.register_agent_type("predator_1", options1)]
                                    #self.agent_type_dict에 다음과 같이 predator의 option을 딕셔너리형태로 저장함.
                                    #{'predator': {'attack_penalty': -0.2, 'attack_range': circle(2), 'hp': 1,
                                    # 'length': 2, 'speed': 1, 'view_range': circle(5), 'width': 2}}
    options2 = {
        "width": 2,
        "length": 2,
        "hp": 1,
        "speed":1,
        "view_range": gw.CircleRange(args.predator2_view_range),
        "attack_range": gw.CircleRange(2),
        "attack_penalty": tag_penalty,
    }
    predator.append(cfg.register_agent_type("predator_2", options2))

    options3 = {
        "width": 1,
        "length": 1,
        "hp": 1,
        "speed": 1.5,
        "view_range": gw.CircleRange(4),
        "attack_range": gw.CircleRange(0),
    }
    prey = [cfg.register_agent_type("prey_1", options3)]
    #prey.append(cfg.register_agent_type("prey_2", options3))



    options4 = {
        "width": 3,
        "length": 3,
        "hp": 0,
        "speed": 1.5,
        "view_range": gw.CircleRange(4),
        "attack_range": gw.CircleRange(0),
    }


    predator_group = [cfg.add_group(predator[0])]       #cfg의 groups 리스트에 predator를 저장하고, 0을 반환한다. predator_group=0
    predator_group.append(cfg.add_group(predator[1]))



    prey_group = [cfg.add_group(prey[0])]  # cfg의 groups 리스트에 predator를 저장하고, 0을 반환한다. predator_group=0




                                                     #a = gw.AgentSymbol(0, index="any")을 넣는 것.
    a_0 = gw.AgentSymbol(predator_group[0], index="any")  #a_0=agent(0,-1)
    a_1 = gw.AgentSymbol(predator_group[1], index="any")  #a_1=agent(1,-1)

    b_0 = gw.AgentSymbol(prey_group[0], index="any")      #b=agent(3,-1)   근데 이건 한번만 실행됨

    cfg.add_reward_rule(gw.Event(a_0, "attack", b_0), receiver=[a_0, b_0], value=[args.tag_reward, -1])
    cfg.add_reward_rule(gw.Event(a_1, "attack", b_0), receiver=[a_1, b_0], value=[args.tag_reward, -1])
    #cfg.add_reward_rule(gw.Event(a_0, "attack", b_0), receiver=[a_0], value=[1, -1])
    # cfg.add_reward_rule(gw.Event(a_1, "attack", b_0), receiver=[a_1], value=[1, -1])

    return cfg

                                #return _parallel_env(
                                #map_size, minimap_mode, env_reward_args, max_cycles, extra_features, render_mode
class _parallel_env(magent_parallel_env, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "hetero_adversarial_v1",
        "render_fps": 1,
    }

    def __init__(
        self,
        map_size,
        minimap_mode,
        reward_args,
        max_cycles,
        extra_features,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            map_size,
            minimap_mode,
            reward_args,
            max_cycles,
            extra_features,
            render_mode,
        )
        assert map_size >= 7, "size of map must be at least 7"
        env = magent2.GridWorld(
            get_config(map_size, minimap_mode, **reward_args), map_size=map_size
        )


        handles = env.get_handles()                                                 #[c_int(0), c_int(1)]
        reward_vals = np.array([1, -1, -1, -1, -1] + list(reward_args.values()))    #[ 1.  -1.  -1.  -1.  -1.  -0.2]
        reward_range = [                                                            #[-4.2, 1.0]
            np.minimum(reward_vals, 0).sum(),
            np.maximum(reward_vals, 0).sum(),
        ]
        names = ["predator_1","predator_2","prey_1"]
                             #중요한 건 이 파트구나.
        super().__init__(    #magent_parallel_env 클래스의 __init__을 실행하는 것!
            env,                #env = magent2.GridWorld(get_config(map_size, minimap_mode, **reward_args), map_size=map_size)
            handles,            #[c_int(0), c_int(1)]
            names,              #names = ["predator", "prey"]
            map_size,           #map_size=45
            max_cycles,         #max_cycles=500
            reward_range,       #[-4.2, 1.0]
            minimap_mode,       #minimap_mode=False
            extra_features,     #extra_features=False
            render_mode,        #render_mode=None
        )

    def generate_map(self):
        env, map_size = self.env, self.map_size
        handles = env.get_handles()  #[c_int(0), c_int(1), c_int(2), c_int(3)]

        #env.add_walls(method="random", n=map_size * map_size * 0.015)
        env.add_walls(method="random", n=0)
        env.add_agents(handles[0], method="random", n=args.n_predator1) #12.6->12개
        env.add_agents(handles[1], method="random", n=args.n_predator2)
        env.add_agents(handles[2], method="random", n=args.n_prey)
        #env.add_agents(handles[3], method="random", n=map_size * map_size * 0.005)  #25.3 ->25개
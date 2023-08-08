import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper 
import imageio
import os

import offline_active_rl.environments
from offline_active_rl.environments.minigrid.wrappers import RGBImgObsWrapper 

env_names = [
    'MiniGrid-Simple-No-Traffic-No-Switch-Red-v0',
    'MiniGrid-Simple-No-Traffic-No-Switch-Confusion-Green-v0',
    'MiniGrid-Simple-Stop-Agent-Switch-v0',
    'MiniGrid-Simple-No-Traffic-No-Switch-v0', # normal env : green with nothing else
]

eval_envs = []
for env_name in env_names:
    eval_env = gym.make(env_name)
    eval_env = FullyObsWrapper(eval_env)
    eval_env = ImgObsWrapper(RGBImgObsWrapper(eval_env))

    # run an episode with random actions and store a gif
    terminate = False
    observations = []

    observations.append(eval_env.reset(seed=123)[0])
    while not terminate:
        action = eval_env.action_space.sample()
        obs, reward, terminate, truncate, info = eval_env.step(action)
        observations.append(obs)
    eval_env.close()

    import ipdb
    ipdb.set_trace()
    # save gif
    if not os.path.exists('./results'):
        os.mkdir('./results')
    imageio.mimsave(f'./results/{env_name}.gif', observations)

    eval_envs.append(eval_env)

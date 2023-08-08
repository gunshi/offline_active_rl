import torch
import numpy as np
import random
import ipdb

from array2gif import write_gif
from typing import Any, Callable, List, Optional
import argparse
import wandb
from PIL import Image

# from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
# from d3rlpy.metrics.scorer import td_error_scorer
# from d3rlpy.metrics.scorer import average_value_estimation_scorer
from offline_active_rl.active_cql import DiscreteActiveCQLConfig
from d3rlpy.dataset import MDPDataset
from d3rlpy.envs import ChannelFirst
import d3rlpy

import gymnasium as gym
import offline_active_rl.environments
from offline_active_rl.environments.minigrid.wrappers import RGBImgObsWrapper 
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper 

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def evaluate_on_benchmark_environments_viz(
    envs: List[gym.Env],
    n_trials: int = 3,
    epsilon: float = 0.0,
    render: bool = False,
    gif: bool = False,
    path: Optional[str] = None,
) -> Callable[..., float]:

    # for image observation
    observation_shape = envs[0].observation_space.shape
    is_image = len(observation_shape) == 3

    frames = []

    def scorer(algo, *args: Any) -> float:
        across_env_reward = []
        rewards_dict = {}
        for env in envs:

            episode_rewards = []
            for trial_itr in range(n_trials):
                observation, _ = env.reset()
                episode_reward = 0.0
                episode_len = 0

                while True:
                    if gif:
                        frames.append(np.moveaxis(env.render(), 2, 0))
                    if np.random.random() < epsilon:
                        action = env.action_space.sample()
                    else:
                        if is_image:
                            action = algo.predict(np.expand_dims(observation, 0))[0]
                        else:
                            action = algo.predict(np.expand_dims(observation, 0))[0]

                    observation, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    episode_len  +=1
    
                    if done:
                        break

                episode_rewards.append(episode_reward)
                print(env.nickname + '- ep len: ', episode_len)
                print(env.nickname + '- ep rew: ', episode_reward)

                if gif:
                    print("Saving gif... ", end="")
                    ipdb.set_trace()
                    write_gif(np.array(frames), 'gifs/' + env.nickname + '_ep_' + str(trial_itr) + ".gif", fps=3)
                    print("Done: ", trial_itr)
                frames.clear()

            # wandb.log(
            #     {
            #         "reward_" + env.nickname: float(np.mean(episode_reward)),
            #     },
            #     step=algo.epoch
            # )
            across_env_reward.append(np.mean(episode_reward)) 
            rewards_dict[env.nickname] = np.mean(episode_reward)
        return rewards_dict

    return scorer

def save_imgs(redstates_dict, redstates_count, step):
    for img_hash in redstates_dict.keys():
        img = Image.fromarray(redstates_dict[img_hash])
        img.save('gifs/runtime/' + str(step) + '_' + str(redstates_count[img_hash]) + '_' + str(img_hash) + ".png")


def main(args):

    data_type = 'expert' # args.dataset_type
    env_data = args.env_data 
    dataset_path = 'data/' + env_data + '/' + data_type
    grid_states = np.load(dataset_path +  '/states.npy')
    rgb_states = np.load(dataset_path +  '/rgb_states.npy')
    actions = np.load(dataset_path +  '/actions.npy')
    rewards = np.load(dataset_path +  '/rewards.npy')
    dones = np.load(dataset_path +  '/dones.npy')

    grid_states_yellow = np.load('data/' + env_data + '/expert_plus_yellow/states.npy')
    rgb_states_yellow = np.load('data/' + env_data + '/expert_plus_yellow/rgb_states.npy')
    actions_yellow = np.load('data/' + env_data + '/expert_plus_yellow/actions.npy')
    rewards_yellow = np.load('data/' + env_data + '/expert_plus_yellow/rewards.npy')
    dones_yellow = np.load('data/' + env_data + '/expert_plus_yellow/dones.npy')

    # try to remove this and see if entirely unnecc
    if True: # no_add_yellow flag was here previously - is this really needed rn? combine these things into single monolithic dataset
        rgb_states = np.concatenate((rgb_states, rgb_states_yellow[18:36]))
        grid_states = np.concatenate((grid_states, grid_states_yellow[18:36]))
        actions = np.concatenate((actions, actions_yellow[18:36].reshape(-1, 1)))
        rewards = np.concatenate((rewards, rewards_yellow[18:36]))
        dones = np.concatenate((dones, dones_yellow[18:36]))

    agent_tl_mask = grid_states[:, 8, 2, 0]==10
    print('agent in light:', agent_tl_mask.sum())
    tl_mask = grid_states[:, :, :, 0]==11.0
    tl_states = grid_states[:, :, :, 1] * tl_mask
    tl_states = tl_states.sum(axis=-1).sum(axis=-1)
    red_mask = (tl_states==0)
    green_mask = (tl_states==1)
    print('reds: ', red_mask.sum())
    print('greens: ', green_mask.sum())

    assert len(grid_states) == len(actions) == len(rewards) == len(dones), "data length mismatch"

    print("states shape: ", grid_states.shape)
    if not args.confused:
        rgb_states[:, 0:8, 0:8] = np.array([100, 100, 100])
    
    states = rgb_states.transpose(0, 3, 1, 2)
    meta_states = grid_states.transpose(0, 3, 1, 2)
    meta_states = np.ascontiguousarray(meta_states)
    states= np.ascontiguousarray(states)

    dataset = MDPDataset(
        states,
        actions,
        rewards,
        dones,
    )


    name = args.wandb_name
    name = name + '_ncr_' + str(args.n_critics)

    wandb.init(
        group=name,
        job_type=str(args.seed),
        project="offline_active_rl",
        entity="causalsampling",
        config=args,
        mode=args.wandb_mode
    )
    wandb.run.name = name + '_seed' + str(args.seed) 
    wandb.run.save()

    set_seed_everywhere(args.seed)
    d3rlpy.seed(args.seed)


    env_names = [
        'MiniGrid-Simple-No-Traffic-No-Switch-Red-v0',
        'MiniGrid-Simple-No-Traffic-No-Switch-Confusion-Green-v0',
        'MiniGrid-Simple-Stop-Agent-Switch-v0',
        'MiniGrid-Simple-No-Traffic-No-Switch-v0', # sanity check normal env : green traffic signal with nothing else
    ]

    eval_envs = []
    for env_name in env_names:

        eval_env = gym.make(env_name)

        eval_env = FullyObsWrapper(eval_env)
        eval_env = ChannelFirst(ImgObsWrapper(RGBImgObsWrapper(eval_env)))
        eval_envs.append(eval_env)

    train_episodes = dataset
    # train_episodes = dataset.episodes # list of episodes with observations of shape (T, 3, 40, 128)
    
    # # save some gifs of the train episodes
    # for ep_id, ep_instance in enumerate(train_episodes):
    #     frames = []
    #     if ep_id > 10:
    #         break
    #     for t in range(len(ep_instance.observations)):
    #         frames.append(ep_instance.observations[t])

    #     write_gif(np.array(frames), 'gifs/' + str(ep_id) + ".gif", fps=3)

    if args.prune_yellow:

        pruned = []
        # yellow_eps = [192, 239, 447, 466, 285, 303, 317, 350, 364, 400, 410, 48, 61, 467, 478, 524, 544]
        # for ep_instance in train_episodes:
        #     if ep_instance.ep_id in yellow_eps:
        #         continue
        #     else:
        #         pruned.append(ep_instance)  
        # train_episodes = pruned
        yellow_eps = [92, 108, 125, 133, 192, 239, 447, 466, 285, 303, 317, 350, 364, 400, 410, 48, 61, 467, 478, 524, 544] #  599, 129
        for ep_instance in train_episodes:
            if ep_instance.ep_id in yellow_eps:
                pruned.append(ep_instance)
            else:
                for i in range(args.data_multiply):
                    pruned.append(ep_instance)
        random.shuffle(pruned)
        train_episodes = pruned

    cql = DiscreteActiveCQLConfig(
        learning_rate=args.lr,
        optim_factory=d3rlpy.models.optimizers.AdamFactory(eps=1e-2 / 32),
        batch_size=args.batch_size,
        alpha=args.alpha_cql,
        n_critics=args.n_critics,
        # encoder_factory='default',
        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
        target_update_interval=args.target_update_interval,
        acq_func=args.acq_func,
        init_rand_sample_epochs=args.init_rand_sample_epochs,
        indep_ensemble=args.indep_ensemble,
        num_bkwds=args.num_bkwds,
    ).create(device=True)


    cql.fit(train_episodes,
            save_interval=args.save_interval,
            n_steps=args.epochs*args.n_steps_per_epoch,
            n_steps_per_epoch=args.n_steps_per_epoch,
            evaluators={
                'environment': evaluate_on_benchmark_environments_viz(eval_envs, gif=args.make_gif),
                # 'advantage': discounted_sum_of_advantage_scorer, # smaller is better
                # 'td_error': td_error_scorer, # smaller is better
                # 'value_scale': average_value_estimation_scorer, # smaller is better
            },
            experiment_name=name + '_seed' + str(args.seed),
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_interval', type=int, default=200)

    parser.add_argument('--batch_size', type=int, default=512)     
    parser.add_argument('--target_update_interval', type=int, default=4)
    parser.add_argument('--n_steps_per_epoch', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument("--lr", default=1e-4, type=float)

    parser.add_argument("--alpha_cql", default=4.0, type=float)
    parser.add_argument('--wandb_mode', choices=['dryrun', 'dryrun_offline', 'online'], default='dryrun')
    parser.add_argument('--wandb_name', default='')
    parser.add_argument("--env_data", default="MiniGrid-Simple-Stop-Light-Rarely-Switch-v0", type=str)

    parser.add_argument('--n_critics', type=int, default=1)

    parser.add_argument("--indep_ensemble", action="store_true", default=False)  
    parser.add_argument("--share_encoder", action="store_true", default=False) 
    parser.add_argument("--prune_yellow", action="store_true", default=False)
    parser.add_argument("--clip_grad", default=1.0, type=float)
    parser.add_argument("--confused", action="store_true", default=False) 
    parser.add_argument("--data_multiply", default=1, type=int)

    parser.add_argument("--init_rand_sample_epochs", default=1, type=int)
    parser.add_argument("--num_bkwds", default=1, type=int)
    parser.add_argument('--acq_func', choices=['random', 'mu_realadv', 'mu_indepadv'], default='random')
    parser.add_argument("--datapath", default="/users/gunpta/code/activesampling/", type=str)    
    parser.add_argument("--make_gif", action="store_true", default=False)

    args = parser.parse_args()
    main(args)


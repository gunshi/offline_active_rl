# Active Sampling in Offline RL

Code for the paper: Can Active Sampling Reduce Causal Confusion in Offline Reinforcement Learning?

Link: https://openreview.net/forum?id=gp2sUQ0uIxx&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3Dcclear.cc%2FCLeaR%2F2023%2FConference%2FAuthors%23your-submissions)


Installation:

pip install -e .
This will also setup the dependencies defined in setup.py

Main Env that this repo is meant to support to start with is the Traffic Grid-World (section 3.2 in paper) - this is set up in the environments folder in this repo and this folder can be copy-pasted into your own repo to use this env on top of  a minigrid install. 
This is a modification of an env used in prior work [1], which constructs the traffic grid-world with a configurable probability for traffic sigal switching between green and red. We modify the dynamics in the env and add spurious correlates to the environment to study causal confusion in offline RL in our paper.
The dataset for offline RL is constructed by rolling out a trained SAC agent in this traffic env, while keeping the red light probability low. In this setup, the randomly instantiated traffic in front of the agent ensures that the dataset has very few episodes where the agent is at the front of a red traffic light. The data can be downloaded at this link, and should be placed inside the repo before running the code.

The code works with the newest repos of minigrid and d3rlpy - both of which are now maintained by Farama-foundation and depend on  instead of gym.

Commands to run active and uniform-sampling versions of CQL:

python traffic_cql_runner.py --acq_func random --n_critics 3 --lr 1e-3 --init_rand_sample_epochs 3

python traffic_cql_runner.py --acq_func mu_realadv --n_critics 3 --lr 1e-3 --init_rand_sample_epochs 3


If you use this code or the traffic world environment or dataset in your research, you can use the following citation:

@inproceedings{
gupta2023can,
title={Can Active Sampling Reduce Causal Confusion in Offline Reinforcement Learning?},
author={Gunshi Gupta and Tim G. J. Rudner and Rowan Thomas McAllister and Adrien Gaidon and Yarin Gal},
booktitle={2nd Conference on Causal Learning and Reasoning},
year={2023},
url={https://openreview.net/forum?id=gp2sUQ0uIxx}
}

[1] Resolving Causal Confusion in Reinforcement Learning via Robust Exploration, Clare Lyle, Amy Zhang, Minqi Jiang, Joelle Pineau, Yarin Gal. ICLR Self-Supervised RL Workshop 2021.

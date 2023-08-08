# Active Sampling in Offline RL

Installation:

pip install -e .
This will also setup the dependencies defined in setup.py

Main Env that this repo is meant to support to start with:
Traffic Grid-World (section 3.2 in paper) - this is set up in the environments folder in this repo and this folder can be copy-pasted into your own repo to use this env on top of  a minigrid install. 

The code works with the neweest repos of minigrid and d3rlpy - both of which are now maintained by Farama-foundation and depend on  instead of gym.

Commands to run active and uniform-sampling versions of CQL.
python traffic_cql_runner.py --acq_func random --n_critics 3 --lr 1e-3 --init_rand_sample_epochs 3
python traffic_cql_runner.py --acq_func mu_realadv --n_critics 3 --lr 1e-3 --init_rand_sample_epochs 3

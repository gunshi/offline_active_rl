from setuptools import setup

setup(name='offline_active_rl',
    version='0.0.1',
    install_requires=(
        'gymnasium',
        'minigrid',
        'imageio',
        'd3rlpy',
        'ipdb',
        'torch',
        'wandb',
        'numpy',
        'array2gif',
        'scipy',
    ),
    packages=['offline_active_rl'],
)
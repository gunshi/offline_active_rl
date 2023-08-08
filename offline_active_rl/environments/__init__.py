from gymnasium.envs.registration import register


register(
    id='MiniGrid-Simple-Stop-Light-8-v0',
    entry_point='offline_active_rl.environments.minigrid:SimpleStopLightEnv8'
)
register(
    id='MiniGrid-Simple-Stop-Light-Green-v0',
    entry_point='offline_active_rl.environments.minigrid:SimpleStopLightEnvGreen'
)
register(
    id='MiniGrid-Simple-No-Traffic-v0',
    entry_point='offline_active_rl.environments.minigrid:SimpleNoTraffic'
)
register(
    id='MiniGrid-Simple-No-Traffic-No-Switch-v0',
    entry_point='offline_active_rl.environments.minigrid:SimpleNoTrafficNoSwitch'
)
register(
    id='MiniGrid-Simple-Stop-Light-Rarely-Switch-v0',                       ##
    entry_point='offline_active_rl.environments.minigrid:SimpleRarelySwitch'
)
register(
    id='MiniGrid-Simple-Stop-Light-Always-Switch-v0',
    entry_point='offline_active_rl.environments.minigrid:SimpleAlwaysSwitch'
)
register(
    id='MiniGrid-Simple-Stop-v0',
    entry_point='offline_active_rl.environments.minigrid:SimpleStop'
)


###################### CC ENVS ##############################

# just red light navigate without yellow square to say 'stop'
register(
    id='MiniGrid-Simple-No-Traffic-No-Switch-Red-v0',
    entry_point='offline_active_rl.environments.minigrid:SimpleNoTrafficNoSwitchRed'
)
# just green light navigate with fixed yellow square to say 'stop' when should be moving
register(
    id='MiniGrid-Simple-No-Traffic-No-Switch-Confusion-Green-v0',
    entry_point='offline_active_rl.environments.minigrid:SimpleNoTrafficNoSwitchConfusedGreen'
)
# switch to red when the agent reaches the green light and yellow square missing because leader moving
register(
    id='MiniGrid-Simple-Stop-Agent-Switch-v0',
    entry_point='offline_active_rl.environments.minigrid:SimpleSwitchForAgent'
)


# spawn before the light, switch it with prob=1.0, yellow square and leading vehicle missing
register(
    id='MiniGrid-Simple-Stop-Sure-Switch-v0',
    entry_point='offline_active_rl.environments.minigrid:SimpleStopSureSwitch'
)
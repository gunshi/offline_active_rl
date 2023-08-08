import gymnasium as gym
from gymnasium import spaces
import operator
from functools import reduce
from offline_active_rl.environments.minigrid.simple_traffic  import OBJECT_TO_IDX, COLOR_TO_IDX


class RGBImgObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as observation,
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.height * env.tile_size, self.env.width * env.tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped
        rgb_img = env.render()

        return {
            'mission': obs['mission'],
            'image': rgb_img
        }


class FullyObsAugWrapper(gym.core.ObservationWrapper):
     """
     Use the image as the only observation output, no language/mission.
     """

     def __init__(self, env):
         super().__init__(env)
         self.observation_spaces["image"] = env.observation_space.spaces['image']
         self.observation_space.spaces["full_image"] = spaces.Box(
             low=0,
             high=255,
             shape=(self.env.width, self.env.height, 3),  # number of cells
             dtype='uint8'
         )

     def observation(self, obs):
         env = self.unwrapped
         full_grid = env.grid.encode()
         full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
             OBJECT_TO_IDX['agent'],
             COLOR_TO_IDX['red'],
             env.agent_dir
         ])

         return {
             'mission': obs['mission'],
             'image': obs['image'],
             'full_image': full_grid,
         }


class FlatImageObsWrapper(gym.core.ObservationWrapper):
     """
     Encode mission strings using a one-hot scheme,
     and combine these with observed images into one flat array
     """

     def __init__(self, env):
         super().__init__(env)

         imgSpace = env.observation_space.spaces['image']
         imgSize = reduce(operator.mul, imgSpace.shape, 1)

         self.observation_space = spaces.Box(
             low=0,
             high=255,
             shape=(imgSize,),
             dtype='uint8'
         )

     def observation(self, obs):
         image = obs['image']

         return image.flatten()

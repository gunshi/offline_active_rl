import numpy as np
from operator import add
import random

from minigrid.core.constants import COLORS, COLOR_TO_IDX, IDX_TO_COLOR 
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Ball, Goal, Key, Lava, Door, Wall, Floor, Box
from minigrid.core.grid import Grid
from minigrid.utils.rendering import fill_coords, point_in_rect
from gymnasium import spaces
from minigrid.core.mission import MissionSpace



# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen'        : 0,
    'empty'         : 1,
    'wall'          : 2,
    'floor'         : 3,
    'door'          : 4,
    'key'           : 5,
    'ball'          : 6,
    'box'           : 7,
    'goal'          : 8,
    'lava'          : 9,
    'agent'         : 10,
    'light'         : 11,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    'open'  : 0,
    'closed': 1,
    'locked': 2,
}


class NewWorldObj:
    """
    Modified version of base class for grid world objects to account for any new object types (Light etc)
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == 'empty' or obj_type == 'unseen':
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == 'wall':
            v = Wall(color)
        elif obj_type == 'floor':
            v = Floor(color)
        elif obj_type == 'ball':
            v = Ball(color)
        elif obj_type == 'key':
            v = Key(color)
        elif obj_type == 'box':
            v = Box(color)
        elif obj_type == 'door':
            v = Door(color, is_open, is_locked)
        elif obj_type == 'goal':
            v = Goal()
        elif obj_type == 'lava':
            v = Lava()
        elif obj_type == 'light':
            v = Light(color)
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError
    

class Light(NewWorldObj):
     def __init__(self, color, contains=None):
         super(Light, self).__init__('light', color)
         self.contains = contains

     def can_pickup(self):
         return False
     def can_contain(self):
         """Can this contain another object?"""
         return True

     def can_overlap(self):
         """Can the agent overlap with this?"""
         return True

     def render(self, img):
         c = COLORS[self.color]

         # Outline
         fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
         fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0,0,0))

         # Horizontal slit
         fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

     def toggle(self, env, pos):
         # Replace the box by its contents
         env.grid.set(*pos, self.contains)
         return True
     

class SimpleStopLightEnv(MiniGridEnv):
    """
    Single-room square grid environment with moving obstacles
    """

    def __init__(
            self,
            size=18,
            agent_start_pos=(1, 2),
            agent_start_dir=0,
            n_obstacles=4,
            init_color='red',
            p_switch=0.1,
            p_switch_back=None,
                                                            ## CC specific features
            rews = {'travel_far': 0.5, 'goal': 5},
            living_cost = 0.001,
            inverse_prob=True,
            irrational_leader=False,
            confused=False,
            respawn_obstacles=True,
            obstacle_skip_prob=0.1,
            fix_confusion=False,
            lone_agent=False,
            nickname='',
            render_mode='rgb_array',
    ):
        self.init_color = init_color
        self.inverse_prob=inverse_prob
        self.irrational_leader=irrational_leader
        self.confused=confused
        self.respawn_obstacles=respawn_obstacles
        self.obstacle_skip_prob = obstacle_skip_prob
        self.fix_confusion = fix_confusion
        self.lone_agent = lone_agent
        self.nickname = nickname

        self.p_switch = p_switch
        self.rews = rews
        self.living_cost = living_cost
        if p_switch_back is None:
            self.p_switch_back = p_switch
        else:
            self.p_switch_back = p_switch_back
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        # Reduce obstacles if there are too many
        if n_obstacles <= size/2 + 1:
            self.n_obstacles = int(n_obstacles)
        else:
            self.n_obstacles = int(size/2)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            width=size, height=5,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
            mission_space=mission_space,
            render_mode=render_mode,
            highlight=False,
            tile_size=8,
        )
        # Allow only 3 actions permitted: left, right, forward
        self.action_space = spaces.Discrete(self.actions.forward + 1)
        self.reward_range = (-1, 1)
        self.max_x = 0

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"
    
    def reset(self, seed=None, options=None):
        self.max_x = 0
        return super().reset(seed=seed, options=options)

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall'          : 'W',
            'floor'         : 'F',
            'door'          : 'D',
            'key'           : 'K',
            'ball'          : 'A',
            'box'           : 'B',
            'goal'          : 'G',
            'lava'          : 'V',
            'light'         : 'L',
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = '_'

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {
            0: '>',
            1: 'V',
            2: '<',
            3: '^'
        }

        str = ''

        for j in range(self.grid.height):

            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue

                c = self.grid.get(i, j)

                if c == None:
                    str += '  '
                    continue

                if c.type == 'door':
                    if c.is_open:
                        str += '__'
                    elif c.is_locked:
                        str += 'L' + c.color[0].upper()
                    else:
                        str += 'D' + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += '\n'

        return str

    def _gen_grid(self, width, height):
        # ipdb.set_trace()

        self.never_been_green = True

        self.made_lone = False
        self.never_color_again = not self.confused
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        self.grid.vert_wall(int(self.width/2), 1, 1)
        self.grid.vert_wall(int(self.width/2), self.height-2, 1)

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Place obstacles
        self.obstacles = []

        obst_ctr = 0
        for i_obst in range(self.n_obstacles):
            if random.random() < 0.1:
                continue
            self.obstacles.append(Ball())
            self.place_obj(self.obstacles[obst_ctr], (i_obst + 2, 2), (1,1), max_tries=100)
            # if obst_ctr==0:
            #     ipdb.set_trace()    
            obst_ctr += 1

        self.mission = "get to the green goal square"

        # traffic light automatically set to a green box
        self.light_colour = self.init_color
        self.light = Light(self.light_colour) 
        self.place_obj(self.light, (int(width/2), 2), (1,1), max_tries = 1)
        # self.place_obj(self.light, (int(width/2), 2) , (1,1), max_tries=1)

        # does an obstacle exist??
        if self.n_obstacles > 0 and len(self.obstacles)>0:
            assert len(self.obstacles)>0, "No obstacles in the environment"

            next_space = self.obstacles[0].cur_pos
            front_cell_leading = self.grid.get(next_space[0]+1, next_space[1])
            not_clear_leading = front_cell_leading and front_cell_leading.type!='light'
            not_clear_leading = not_clear_leading or ( front_cell_leading and front_cell_leading.type == 'light' and self.light_colour=='red')
            not_clear_leading = not_clear_leading or (next_space[0] - self.agent_pos[0])<2
            not_clear_leading = not_clear_leading and (not self.never_color_again)

            # color square based on not-clear_leading
            if not_clear_leading:
                self.grid.get(0, 0).color='yellow'
            else:
                self.grid.get(0, 0).color='grey'

        self.timer = 0

        if self.fix_confusion:
            self.grid.get(0, 0).color='yellow'

    def step(self, action):            
        # Invalid action
        self.timer +=1
        if self.inverse_prob:
            if self.timer<7:
                now_p_switch=self.p_switch
            elif self.timer<13:
                now_p_switch=self.p_switch/10.0
            else:
                now_p_switch=self.p_switch
        else:
            now_p_switch = self.p_switch
            
        if action >= self.action_space.n:
            action = 0

        # Check if there is an obstacle in front of the agent
        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type != 'goal' and front_cell.type!='light'
        not_clear = not_clear or ( front_cell and front_cell.type == 'light' and self.light_colour=='red')

        # Update obstacle positions
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            top = tuple(map(add, old_pos, (1, 0)))
            if top[0] == self.light.cur_pos[0]:
                if self.light_colour == 'green':
                    top = (top[0] + 1, top[1])
            if top[0] == self.width-1:
                top = (1, old_pos[1])
                try:
                    if self.respawn_obstacles:
                        self.place_obj(self.obstacles[i_obst], top=top, size=(2,1), max_tries=100)
                    self.grid.set(*old_pos, None)
                    if i_obst==0:
                        self.never_color_again = True
                except:
                    pass
            else:
                try:
                    self.place_obj(self.obstacles[i_obst], top=top, size=(1,1), max_tries=100)
                    self.grid.set(*old_pos, None)
                except:
                    pass
    

        # Light changes colour with some probability       
        if self.lone_agent and (self.light.cur_pos[0]-self.agent_pos[0] ==2) \
            and (self.light.cur_pos[1]==self.agent_pos[1]) and not self.made_lone: 
            if self.light_colour == 'green':
                self.made_lone = True
                self.light_colour = 'red' 
                self.light.color = self.light_colour 
                light_pos = self.light.cur_pos 
                self.grid.set(*light_pos, None)
                self.place_obj(self.light, top=light_pos, size=(1,1), max_tries=100)

        else:
            p_switch = now_p_switch if self.light.color == 'green' else self.p_switch_back
            if np.random.rand() > 1 - p_switch and self.agent_pos[0] != self.light.cur_pos[0]:

                front_cell = self.grid.get(*self.front_pos)

                p_agentnear_block = 1.0/(2*(self.light.cur_pos[0]-self.agent_pos[0]))
                if (self.light_colour=='green' \
                    and self.agent_pos[0]<self.light.cur_pos[0] \
                        and self.inverse_prob \
                            and np.random.rand() > 1 - p_agentnear_block) or (self.light_colour == 'red' \
                        and (self.nickname=='simplered' or self.nickname=='switchforagent') \
                        and self.timer<30):
                    pass
                else:
                    self.light_colour = 'red' if self.light_colour == 'green' else 'green'
                    if self.light_colour == 'green':
                        self.never_been_green = False
                    self.light.color = self.light_colour 
                    light_pos = self.light.cur_pos 
                    self.grid.set(*light_pos, None)
                    self.place_obj(self.light, top=light_pos, size=(1,1), max_tries=100)
            else:
                if self.nickname=='simplered' and self.timer>150 and self.never_been_green:
                    assert self.light_colour == 'red'
                    print('INTERVENING')
                    self.light_colour = 'green'
                    self.never_been_green = False
                    self.light.color = self.light_colour 
                    light_pos = self.light.cur_pos 
                    self.grid.set(*light_pos, None)
                    self.place_obj(self.light, top=light_pos, size=(1,1), max_tries=100)


        if self.n_obstacles > 0 and len(self.obstacles)>0:

            # Check if there is an obstacle in front of the leading vehicle
            next_space = self.obstacles[0].cur_pos
            front_cell_leading = self.grid.get(next_space[0]+1, next_space[1])
            not_clear_leading = front_cell_leading and front_cell_leading.type!='light' and front_cell_leading.type!='wall'
            not_clear_leading = not_clear_leading or ( front_cell_leading and front_cell_leading.type == 'light' and self.light_colour=='red')
            not_clear_leading = not_clear_leading or (next_space[0] - self.agent_pos[0])<2
            not_clear_leading = not_clear_leading and (not self.never_color_again)


            # color square based on not-clear_leading
            if not_clear_leading:
                self.grid.get(0, 0).color='yellow'
            else:
                self.grid.get(0, 0).color='grey'

        # Update the agent's position/direction
        obs, reward, terminated, truncated, info = MiniGridEnv.step(self, action)
        done = terminated

        if reward > 0.9 and self.rews['goal']>0:
            reward = self.rews['goal']

        if reward == 0:
            # reward = 0.01 * (self.front_pos[0] / self.width)**2

            if self.front_pos[0] > self.max_x:
                reward = self.rews['travel_far'] * self.front_pos[0] / self.width
                self.max_x = self.front_pos[0]

        # If the agent tried to walk over an obstacle or wall
        if action == self.actions.forward and not_clear:
            reward = -1
            done = True
            return obs, reward, done, truncated, info
        
        if self.nickname=='simplegreen' and self.timer>30:
            done = True
            reward = -1
            return obs, reward, done, truncated, info

        reward = reward - self.living_cost
        return obs, reward, done, truncated, info




class SimpleStopLightEnv8(SimpleStopLightEnv):
    def __init__(self):
        super().__init__(size=16, n_obstacles=3, rews={'travel_far': 0, 'goal': 0})

class SimpleStopLightEnvGreen(SimpleStopLightEnv):
    def __init__(self):
        super().__init__(size=16, n_obstacles=3, init_color='green', p_switch=0.1, rews={'travel_far': 0, 'goal': 0})

class SimpleNoTraffic(SimpleStopLightEnv): # change this back
    def __init__(self):
        super().__init__(size=16, n_obstacles=0, init_color='green', p_switch=0.1, rews={'travel_far': 0, 'goal': 0}) 
        # super().__init__(size=16, n_obstacles=0, init_color='green', p_switch=0.05, rews={'travel_far': 0, 'goal': 0}) 

# green with no switching
class SimpleNoTrafficNoSwitch(SimpleStopLightEnv):
    def __init__(self):
        super().__init__(size=16,
                        n_obstacles=0,
                        init_color='green',
                        p_switch=0.0,
                        p_switch_back=0.0,
                        rews={'travel_far': 0, 'goal': 0},
                        nickname='green_notrafficnoswitch',
        ) 

# spawn at stop light, with some switch prob
class SimpleStop(SimpleStopLightEnv):
    def __init__(self):
        super().__init__(size=16,
                        n_obstacles=3,
                        init_color='green',
                        p_switch=0.1,
                        agent_start_pos=(7, 2),
                        rews={'travel_far': 0, 'goal': 0},
                        nickname='stop',        
                ) 

# green with rare switching
class SimpleRarelySwitch(SimpleStopLightEnv):
    def __init__(self):
        super().__init__(size=16,
                        n_obstacles=4,
                        init_color='green',
                        p_switch=0.03, p_switch_back=0.3, 
                        ews={'travel_far': 0, 'goal': 0},
                        nickname='rarelyswitch',
                        ) 

# Hard env to find out that you c
class SimpleAlwaysSwitch(SimpleStopLightEnv):
    def __init__(self):
        super().__init__(size=16,
                        n_obstacles=4,
                        init_color='green',
                        p_switch=0.5,
                        rews={'travel_far': 0, 'goal': 0},
                        nickname='alwaysswitch',
                        inverse_prob=False) 




class SimpleSwitchForAgent(SimpleStopLightEnv):
    def __init__(self):
        super().__init__(size=16,
                        n_obstacles=3, 
                        init_color='green',
                        p_switch=0.0,
                        p_switch_back=0.02,
                        rews={'travel_far': 0, 'goal': 0},
                        confused=True,
                        lone_agent=True,
                        respawn_obstacles=False,
                        nickname='switchforagent',
                        inverse_prob=False,
        ) 
class SimpleNoTrafficNoSwitchRed(SimpleStopLightEnv):
    def __init__(self):
        super().__init__(
            size=16, n_obstacles=0, init_color='red', p_switch=0.00, p_switch_back=0.02,
            rews={'travel_far': 0, 'goal': 0},
            confused=False,
            nickname='simplered',
            inverse_prob=False,
        ) 
class SimpleNoTrafficNoSwitchConfusedGreen(SimpleStopLightEnv):
    def __init__(self):
        super().__init__(
            size=16, n_obstacles=0, init_color='green', p_switch=0.0, p_switch_back=0.0,
            rews={'travel_far': 0, 'goal': 0},
            confused=True, 
            fix_confusion=True, 
            nickname='simplegreen'
        ) 


class SimpleStopSureSwitch(SimpleStopLightEnv):
    def __init__(self):
        super().__init__(
                        size=16, n_obstacles=3, 
                        init_color='green', p_switch=1.0, p_switch_back=0.09,
                        agent_start_pos=(6, 2), rews={'travel_far': 0, 'goal': 0},
                        confused=False, inverse_prob=False, nickname='stopsurenoyellow'
                ) 



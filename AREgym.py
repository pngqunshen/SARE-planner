import gym
import numpy as np
import math
import World
import cv2
import utils
'''
    Observations: 36 range values from lidar, 36 heuristics for laserscan value
    
    Action: 1 continuous value for heading 
    receives from 0-1, scaled to -pi to pi. 0 is +x direction (down)

    Reward: ???

    World axis:
    based on indexing of 2d array 
    ----> y
    |
    |
    V
    x

    0 : obstacle
    1 : free space
    -1: unexplored
'''

class AREEnv(gym.Env):
    def __init__(self, grid, step_distance):
        self.observation_size = 72
        self.world_size = grid # how large to generate the square map, in pixels
        self.step_distance = step_distance # how far the robot moves at each time step, in pixels

        self.world_generator = World.WorldGenerator()
        self.world = self.world_generator.new_world(self.world_size)
        
        # initialize all unexplored areas as -1
        self.global_map = np.zeros((2 * grid, 2 * grid)) - 1

        # initialize robot in centre of global map
        self.x = grid
        self.y = grid 

        # initialize lidar
        self.laser_scan_max_dist = 50 # in pixels
        self.num_laser_scan = int(self.observation_size // 2)
        self.num_laser_scan = 36
        self.laserscan = np.zeros(self.num_laser_scan)
        # self.laser_scan_heading = np.array([0])
        self.laser_scan_heading = np.array([((0.5 - (i / self.num_laser_scan)) * math.pi * 2) for i in range(self.num_laser_scan)])


        # initialize heuristics and heuristics params
        self.heuristic = np.zeros((self.num_laser_scan))
        self.heuristic_dist = self.step_distance * 2 # magic number 

        # episode termination criteria param
        self.termination_threshhold = 0.9 ## also magic number

        # keeping track of stuff for rendering 
        self.scan = [] # store values scanned pixels for rendering
        self.total_steps = []
        self.initial_x = self.x
        self.initial_y = self.y

    def reset(self):
        self.world = self.world_generator.new_world(self.world_size)
        self.x = self.world_size
        self.y = self.world_size

        # keeping track of stuff for rendering 
        self.scan = [] # store values scanned pixels for rendering
        self.total_steps = []
        self.initial_x = self.x
        self.initial_y = self.y

        return self.observe()

    def step(self, action):
        # scale action between -pi and pi
        heading = (action - 0.5) * math.pi * 2
        
        steps = self.world.get_path(heading, self.step_distance)

        for step in steps:
            self.move(step[0], step[1])
            self.get_laser_scan()

            # keep track for render
            self.total_steps.append(step)
        return self.observe(), self.get_reward(), self.finished()
    
    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        self.world.move(dx, dy)
    
    def observe(self):
        self.get_laser_scan()
        self.calc_heuristics()

        # scale laserscan between 1 and 0
        scaled_scan = self.laserscan / self.laser_scan_max_dist
        observation = np.concatenate((scaled_scan, self.heuristic))
        return observation

    def get_laser_scan(self):
        self.scan.clear()

        for i in range(self.num_laser_scan):
            distance = 1
            dx = 0
            dy = 0
            # ensure that current state is explored, only used when called during reset
            self.global_map[self.x + dx, self.y +dy] = 1

            heading = self.laser_scan_heading[i]
            while distance < self.laser_scan_max_dist:
                # move in direction until hit an obstacle or reach maximum distance

                dx = int(math.cos(heading) * distance)
                dy = int(math.sin(heading) * distance)
                if not utils.out_of_bounds(self.world.x + dx, self.world.y + dy, self.world.size):
                    break

                if self.world.world[self.world.x + dx, self.world.y + dy] == 0:
                    self.global_map[self.x + dx, self.y +dy] = 0
                    break
                # update robot's map
                self.global_map[self.x + dx, self.y +dy] = 1

                # update world, used for calculation of termination
                self.world.explore(dx, dy)

                # for rendering
                self.scan.append([dx,dy])
                distance += 1
            self.laserscan[i] = distance # in pixels
            
    def calc_heuristics(self):
        for i in range(self.num_laser_scan):
            '''
            ███╗░░░███╗░█████╗░░██████╗░██╗░█████╗░
            ████╗░████║██╔══██╗██╔════╝░██║██╔══██╗
            ██╔████╔██║███████║██║░░██╗░██║██║░░╚═╝
            ██║╚██╔╝██║██╔══██║██║░░╚██╗██║██║░░██╗
            ██║░╚═╝░██║██║░░██║╚██████╔╝██║╚█████╔╝
            ╚═╝░░░░░╚═╝╚═╝░░╚═╝░╚═════╝░╚═╝░╚════╝░
            '''

            self.heuristic[i] = 1
        
    def get_reward(self):
        '''
        ███╗░░██╗░█████╗░   ░█████╗░██╗░░░░░██╗░░░██╗███████╗
        ████╗░██║██╔══██╗   ██╔══██╗██║░░░░░██║░░░██║██╔════╝
        ██╔██╗██║██║░░██║   ██║░░╚═╝██║░░░░░██║░░░██║█████╗░░
        ██║╚████║██║░░██║   ██║░░██╗██║░░░░░██║░░░██║██╔══╝░░
        ██║░╚███║╚█████╔╝   ╚█████╔╝███████╗╚██████╔╝███████╗
        ╚═╝░░╚══╝░╚════╝░   ░╚════╝░╚══════╝░╚═════╝░╚══════╝
        '''
        return 1

    def finished(self):
        return self.world.explore_progress() >= self.termination_threshhold

    '''
    IDK HOW WELL THIS RENDERING WORKS, JUST USING FOR CHECKING NOW

    global: show global map
    world: show world map, not including robot, can use used for verifying world generation
    default: global map with robot and laserscan

    [Blue Green Red]
    unexplored: grey [128, 128, 128]
    obstacle: black [0, 0, 0]
    free: white [255, 255, 255]
    laser: pink [255, 0, 255]
    robot: teal [255, 255, 0]
    path: blue [255, 255, 0]
    '''
    def render(self, key = ""):
        if key == "global":
            img = np.zeros((self.world_size * 2, self.world_size * 2, 3))
            for i in range(self.world_size * 2):
                for j in range(self.world_size * 2):
                    img[i,j,0] = 0 if self.global_map[i,j] == 0 else 1 if self.global_map[i,j] == 1 else 0.5
                    img[i,j,1] = 0 if self.global_map[i,j] == 0 else 1 if self.global_map[i,j] == 1 else 0.5
                    img[i,j,2] = 0 if self.global_map[i,j] == 0 else 1 if self.global_map[i,j] == 1 else 0.5

            x, y = self.initial_x, self.initial_y
            for steps in self.total_steps:
                img[x, y, 1] = 0
                img[x, y, 2] = 0
                x += steps[0]
                y += steps[1]


        elif key == "world":
            img = np.zeros((self.world_size, self.world_size, 3))
            for i in range(self.world_size):
                for j in range(self.world_size):
                    img[i,j,0] = 0 if self.world.world[i,j] == 0 else 1
                    img[i,j,1] = 0 if self.world.world[i,j] == 0 else 1
                    img[i,j,2] = 0 if self.world.world[i,j] == 0 else 1
        else:
            img = np.zeros((self.world_size, self.world_size, 3))
            for i in range(self.world_size):
                for j in range(self.world_size):
                    img[i,j,0] = 0 if self.world.world[i,j] == 0 else 1
                    img[i,j,1] = 0 if self.world.world[i,j] == 0 else 1
                    img[i,j,2] = 0 if self.world.world[i,j] == 0 else 1
            img[self.world.x, self.world.y, 2] = 0
            print(self.world.x)
            print(self.world.y)
            for i in range(len(self.scan)):
                img[self.world.x + self.scan[i][0], self.world.y + self.scan[i][1], 1] = 0
            
        ## to align with numpy
        # x is down 
        # y is right
        image = cv2.rotate(img, cv2.ROTATE_180)
        cv2.imshow("image", image)
        cv2.waitKey()
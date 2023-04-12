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
    def __init__(self, grid, step_distance, save_map=False, laser_scan_max_dist=50, \
                 num_laser_scan=36, heuristic_dist=50, termination_threshhold=0.9):
        self.world_size = grid # how large to generate the square map, in pixels
        self.step_distance = step_distance # how far the robot moves at each time step, in pixels

        self.world_generator = World.WorldGenerator(self.world_size, self.world_size)
        self.world = self.world_generator.new_world()
        
        # initialize all unexplored areas as -1
        self.global_map = np.zeros((2 * grid, 2 * grid)) - 1

        # initialize robot in centre of global map
        self.x = grid
        self.y = grid 

        # initialize lidar
        self.laser_scan_max_dist = laser_scan_max_dist # in pixels
        self.num_laser_scan = num_laser_scan
        self.laserscan = np.zeros(self.num_laser_scan)
        self.laser_scan_heading = \
            np.array([(utils.action_to_rad(i / self.num_laser_scan)) \
                      for i in range(self.num_laser_scan)])


        # initialize heuristics and heuristics params
        self.heuristic = np.zeros((self.num_laser_scan))
        self.heuristic_dist = heuristic_dist

        # initialize reward 
        self.reward_prev = 0
        self.reward_after = 0

        # episode termination criteria param
        self.termination_threshhold = termination_threshhold ## also magic number

        self.save_map = save_map
        if self.save_map:
            self.save_map_mode()
    
    def save_map_mode(self):
        '''
        global: show global map
        world: show world map, not including robot, can use used for 
        verifying world generation
        default: global map with robot and laserscan

        [Blue Green Red]
        unexplored: grey [128, 128, 128]
        obstacle: black [0, 0, 0]
        free: white [255, 255, 255]
        laser: pink [255, 0, 255]
        robot: teal [255, 255, 0]
        path: blue [255, 255, 0]
        '''
        
        # shows global map and path taken so far
        self.map_img = np.zeros((self.world_size * 2, self.world_size * 2, 3))
    
        # image of generated map, doesnt change 
        self.world_img = np.ones((self.world_size, self.world_size, 3))
        for i in range(self.world_size):
            for j in range(self.world_size):
                self.world_img[i,j,0] = 0 if self.world.world[i,j] == 0 else 1
                self.world_img[i,j,1] = 0 if self.world.world[i,j] == 0 else 1
                self.world_img[i,j,2] = 0 if self.world.world[i,j] == 0 else 1

        # shows world map with Robot and curent laserscan and path taken
        self.current_img = np.zeros((self.world_size, self.world_size, 3))
        for i in range(self.world_size):
            for j in range(self.world_size):
                self.current_img[i,j,0] = 0 if self.world.world[i,j] == 0 else 1
                self.current_img[i,j,1] = 0 if self.world.world[i,j] == 0 else 1
                self.current_img[i,j,2] = 0 if self.world.world[i,j] == 0 else 1

        # keeping track of stuff for rendering 
        self.scan = [] # store values scanned pixels for rendering

    # reset and generate a new world
    def reset(self):
        self.world = self.world_generator.new_world()
        self.x = self.world_size
        self.y = self.world_size
        self.global_map = np.zeros((2 * self.world_size, 2 * self.world_size)) - 1

        # keeping track of stuff for rendering 
        self.scan = [] # store values scanned pixels for rendering
        self.total_steps = []
        self.initial_x = self.x
        self.initial_y = self.y

        if self.save_map:
            self.save_map_mode()

        return self.observe()
    
    '''
    Take one step in action [0,1] direction
    Updates rewards and map for rendering
    Calls the move function to actually move the bot in global and world
    '''
    def step(self, action):
        # scale action between -pi and pi
        heading = utils.action_to_rad(action)

        # may have to change step_distance to laser scan distance        
        steps = self.world.get_path(heading, self.step_distance)
        
        self.reward_prev = self.world.explore_progress()

        for step in steps:
            self.move(step[0], step[1])
            self.get_laser_scan()

            if self.save_map:
                self.map_img[self.x, self.y, 0] = 1
        
        self.reward_after = self.world.explore_progress()

        # runnning slowly so i commented
        return self.observe(), self.get_reward(), self.finished()
    
    '''
    Move the bot in dx and dy direction in global map
    Calls the move function in world to physically move the bot
    '''
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
    
    '''
    Take the laser scan at current point based on world map
    Updates self.laserscan
    '''
    def get_laser_scan(self):
        if self.save_map:
            self.scan.clear()

        for i in range(self.num_laser_scan):
            heading = self.laser_scan_heading[i]
            # function to update in bresenham
            def update_func(dx, dy):
                self.global_map[self.x + dx, self.y + dy] = 1
                if self.world.world[self.world.x + dx, self.world.y + dy] == 0:
                    self.global_map[self.x + dx, self.y + dy] = 0
                    return
                self.world.explore(dx, dy)
                if self.save_map:
                    self.scan.append([dx,dy])
            # terminating condition in bresenham
            def term_cond(dx, dy):
                return  utils.out_of_bounds(self.world.x + dx, self.world.y + dy, \
                                            self.world.size, self.world.size) or \
                        self.world.world[self.world.x + dx, self.world.y + dy] == 0
            x1, y1 = utils.bresenham_line(lambda dx, dy: update_func(dx, dy), \
                                          lambda dx, dy: term_cond(dx, dy), \
                                          self.x, self.y, self.laser_scan_max_dist, \
                                          heading)
            self.laserscan[i] = utils.euc_dist(self.x,x1,self.y,y1) # in pixels

    '''
    Calculate heuristic
    Do a laserscan from the current point to find the furthest possible point
    Heuristic at each heading is the total total unexplored area from another
    laser scan at that point (using heuristic distance), scaled to [0,1]
    '''      
    def calc_heuristics(self):
        heu = np.zeros((self.num_laser_scan))
        for i in range(self.num_laser_scan):
            heading = self.laser_scan_heading[i]
            def update_func(dx, dy):
                pass
            def term_cond(dx, dy):
                return utils.out_of_bounds(self.x + dx, self.y + dy, \
                                           self.world_size, self.world_size) or \
                        self.global_map[self.x + dx, self.y + dy] != 1
            x1, y1 = utils.bresenham_line(lambda dx, dy: update_func(dx, dy), \
                                          lambda dx, dy: term_cond(dx, dy), \
                                          self.x, self.y, self.laser_scan_max_dist, \
                                          heading)
            area_to_explore = 0
            for j in range(self.num_laser_scan):
                heading2 = self.laser_scan_heading[j]
                can_explore = np.zeros((2 * self.world_size, 2 * self.world_size))
                def update_func2(dx, dy):
                    if self.global_map[x1 + dx, y1 + dy] == -1:
                        can_explore[x1 + dx, y1 + dy] = 1
                def term_cond2(dx, dy):
                    return  utils.out_of_bounds(x1 + dx, y1 + dy, \
                                                self.world_size, self.world_size) or \
                            self.global_map[x1 + dx, y1 + dy] == 0
                x2, y2 = utils.bresenham_line(lambda dx, dy: update_func2(dx, dy), \
                                              lambda dx, dy: term_cond2(dx, dy), \
                                              x1, y1, self.heuristic_dist, \
                                              heading2)
                area_to_explore += can_explore.sum()
            heu[i] = area_to_explore
        self.heuristic = heu/max(heu.max(),1)
        
    def get_reward(self):
        '''
        ███╗░░██╗░█████╗░   ░█████╗░██╗░░░░░██╗░░░██╗███████╗
        ████╗░██║██╔══██╗   ██╔══██╗██║░░░░░██║░░░██║██╔════╝
        ██╔██╗██║██║░░██║   ██║░░╚═╝██║░░░░░██║░░░██║█████╗░░
        ██║╚████║██║░░██║   ██║░░██╗██║░░░░░██║░░░██║██╔══╝░░
        ██║░╚███║╚█████╔╝   ╚█████╔╝███████╗╚██████╔╝███████╗
        ╚═╝░░╚══╝░╚════╝░   ░╚════╝░╚══════╝░╚═════╝░╚══════╝
        '''
        # Calculate the delta of exploration progress before and after the action
        delta_reward = self.reward_after - self.reward_prev

        '''        
        ██╗  ████████╗██╗░░██╗██╗███╗░░██╗██╗░░██╗  ██╗████████╗██╗░██████╗
        ██║  ╚══██╔══╝██║░░██║██║████╗░██║██║░██╔╝  ██║╚══██╔══╝╚█║██╔════╝
        ██║  ░░░██║░░░███████║██║██╔██╗██║█████═╝░  ██║░░░██║░░░░╚╝╚█████╗░
        ██║  ░░░██║░░░██╔══██║██║██║╚████║██╔═██╗░  ██║░░░██║░░░░░░░╚═══██╗
        ██║  ░░░██║░░░██║░░██║██║██║░╚███║██║░╚██╗  ██║░░░██║░░░░░░██████╔╝
        ╚═╝  ░░░╚═╝░░░╚═╝░░╚═╝╚═╝╚═╝░░╚══╝╚═╝░░╚═╝  ╚═╝░░░╚═╝░░░░░░╚═════╝░

        ██████╗░░█████╗░███╗░░██╗███████╗
        ██╔══██╗██╔══██╗████╗░██║██╔════╝
        ██║░░██║██║░░██║██╔██╗██║█████╗░░
        ██║░░██║██║░░██║██║╚████║██╔══╝░░
        ██████╔╝╚█████╔╝██║░╚███║███████╗
        ╚═════╝░░╚════╝░╚═╝░░╚══╝╚══════╝
        '''
        return delta_reward

    def finished(self):
        return self.world.explore_progress() >= self.termination_threshhold
    
    '''
    Magical rendering function that spits out images
    '''
    def render(self, image_path=""):
        
        if not self.save_map:
            return
        
        for i in range(self.world_size * 2):
            for j in range(self.world_size * 2):
                if (self.map_img[i,j] == np.array([1 ,0, 0])).all():
                    continue
                self.map_img[i,j,0] = 0 \
                    if self.global_map[i,j] == 0 \
                    else 1 \
                        if self.global_map[i,j] == 1 \
                        else 0.5
                self.map_img[i,j,1] = 0 \
                    if self.global_map[i,j] == 0 \
                    else 1 \
                        if self.global_map[i,j] == 1 \
                        else 0.5
                self.map_img[i,j,2] = 0 \
                    if self.global_map[i,j] == 0 \
                    else 1 \
                        if self.global_map[i,j] == 1 \
                        else 0.5
        image1 = cv2.rotate(self.map_img, cv2.ROTATE_180)
        image1 = (image1 * 255).astype(np.uint8)
        cv2.imwrite(image_path + "/global_map.png", image1)

        image2 = cv2.rotate(self.world_img, cv2.ROTATE_180)
        image2 = (image2 * 255).astype(np.uint8)
        cv2.imwrite(image_path + "/world_map.png", image2)

        self.current_img[self.world.x, self.world.y, 2] = 0
        for i in range(len(self.scan)):
            self.current_img[self.world.x + self.scan[i][0], \
                             self.world.y + self.scan[i][1], 1] = 0
        image3 = cv2.rotate(self.current_img, cv2.ROTATE_180)
        image3 = (image3 * 255).astype(np.uint8)
        cv2.imwrite(image_path + "/current_state.png", image3)

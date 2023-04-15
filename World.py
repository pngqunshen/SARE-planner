import numpy as np
import math
import utils


class World:
    def __init__(self, world, x, y):
        self.world = world
        self.size = len(world[1])
        self.x = x
        self.y = y

        # total exploraable area
        self.free_area = self.world.sum()
        self.explored_area = np.zeros_like(self.world)

    def get_path(self, heading, max_distance):
        # this returns a list of steps to take in a form of [dx, dy] between each step
        laser_steps = np.arange(max_distance)
        x_laser = np.cos(heading)
        y_laser = np.sin(heading)
        x_world = ((x_laser * laser_steps).astype(int) + self.x).clip(0,self.size-1)
        y_world = ((y_laser * laser_steps).astype(int) + self.y).clip(0,self.size-1)
        term = self.world[x_world, y_world] == 0 # true if  obstacle in real world
        term1 = np.argmax(term, axis=0) # find first row that is true (first obstacle)
        term2 = np.all(term == False, axis=0) # find rows without obstacles
        if term2:
            term1 = max_distance # rows without obstacles scan all the way
        ind_free = (np.arange(max_distance) < term1)
        path = np.hstack(((x_world-self.x)[ind_free].reshape(-1,1), (y_world-self.y)[ind_free].reshape(-1,1)))
        path_new = np.zeros_like(path)
        path_new[0] = path[0]
        path_new[1:] = path[1:] - path[:-1]
        # create a boolean mask that is True for all rows that do not contain [0, 0]
        mask = np.any(path_new != [0, 0], axis=1)
        return path_new[mask]

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
    
    def explore(self, dx, dy):
        self.explored_area[self.x + dx, self.y + dy] = 1

    def explore_progress(self):
        return self.explored_area.sum() / self.free_area

class WorldGenerator:
    def __init__(self, size_x, size_y, fill = 0.5):
        self.size_x = size_x # x size of world
        self.size_y = size_y # y size of world
        self.fill = fill # percentage [0,1] of world that is filled with room

    def new_world(self):
        '''
        Generate random world and randomly place robot in world
        '''
        world = np.zeros((self.size_x, self.size_y))

        num_rooms = 10 # number of large squares initially
        min_adjacent_px = self.size_x//20 # number of pixels to align

        room_area = self.fill * self.size_x * self.size_y / num_rooms
        room_size = int(math.sqrt(room_area))
        first_room = True
        
        while num_rooms > 0:
            curr_iter = 0 # to keep track of number of tries to fit a room
            while True:
                # try to fit 100000 times
                if curr_iter > 100000 and room_size > 1 and min_adjacent_px > 1:
                    room_size //= 2 # smaller room so can fit
                    num_rooms *= 4 # need more rooms to fill area
                    min_adjacent_px //= 2 # scale with room_size
                    curr_iter = 0
                # choose a random point that can generate a room of room_area
                start = np.random.randint(1,[self.size_x-room_size, self.size_y-room_size])
                if first_room or self.can_gen_room(world,room_size,start[0],start[1],min_adjacent_px):
                    self.flood_room(world,room_size,start[0],start[1])
                    first_room = False
                    break
                curr_iter += 1
            num_rooms -= 1
        while True:
            robot_start = np.random.randint(0,[self.size_x, self.size_y])
            if world[robot_start[0],robot_start[1]] == 1:
                break
        return World(world, robot_start[0], robot_start[1])
    
    def can_gen_room(self, world, room_size, room_x, room_y, min_adjacent_px=1): 
        # at least 1 connecting px to adjacent room
        return not world[room_x:room_x+room_size,room_y:room_y+room_size].any() and (\
            (room_y>0 and \
             world[room_x:room_x+room_size,room_y-1].sum()>min_adjacent_px) or \
            (room_y+room_size<self.size_y-1 and \
             world[room_x:room_x+room_size,room_y+room_size].sum()>min_adjacent_px) or \
            (room_x>0 and \
             world[room_x-1,room_y:room_y+room_size].sum()>min_adjacent_px) or \
            (room_x+room_size<self.size_x-1 and \
             world[room_x+room_size,room_y:room_y+room_size].sum()>min_adjacent_px))

    def flood_room(self, world, room_size, room_x, room_y):
        world[room_x:room_x+room_size,room_y:room_y+room_size] = 1

        
# not square rooms, but rectangles. Semi working, but haven't integrated with OOP yet. It's just it's own function now. 

import numpy as np
import random

class WorldGenerator:
    def __init__(self, size_x, size_y, fill=0.5, min_adj_px=1):
        self.size_x = size_x
        self.size_y = size_y
        self.fill = fill
        self.min_adj_px = min_adj_px
        
    def generate(self, num_rooms=10, min_size=50, max_size=100):
        world = np.zeros((self.size_x, self.size_y))
        
        while np.sum(world) / (self.size_x * self.size_y) < self.fill:
            room_x = random.randint(min_size, max_size)
            room_y = random.randint(min_size, max_size)
            
            pos_x = random.randint(0, self.size_x - room_x)
            pos_y = random.randint(0, self.size_y - room_y)
        
            if np.any(world[pos_x:pos_x+room_x, pos_y:pos_y+room_y]) and world[pos_x, pos_y] == 1:
                world[pos_x:pos_x+room_x, pos_y:pos_y+room_y] = 1
            
            world[pos_x:pos_x+room_x, pos_y:pos_y+room_y] = 1
        
        return world


import matplotlib.pyplot as plt

world_gen = WorldGenerator(500, 500)
world = world_gen.generate()

plt.imshow(world, cmap='gray')
plt.show()

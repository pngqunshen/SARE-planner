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
        distance = 0
        dx = 0
        dy = 0

        steps = [[0,0]]
        dx_prev = 0
        dy_prev = 0

        while distance < max_distance:
            # move in direction until hit an obstacle or reach maximum distance

            dx = int(math.cos(heading) * distance)
            dy = int(math.sin(heading) * distance)
            # check out of bounds
            if not utils.out_of_bounds(self.x + dx, self.y + dy, self.size):
                break
            if self.world[self.x + dx, self.y + dy] == 0:
                break        
            steps.append([dx - dx_prev, dy - dy_prev])
            dx_prev, dy_prev = dx, dy
            distance += 1

        return steps

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
    
    def explore(self, dx, dy):
        self.explored_area[self.x + dx, self.y + dy] = 1

    def explore_progress(self):
        return self.explored_area.sum() / self.free_area

# class WorldGenerator:
#     def __init__(self):
#         return

#     def new_world(self, grid):
#         '''
#         Generate random world and randomly place robot in world
#         '''
#         world = np.zeros((grid,grid))

#         for i in range(125,375):
#             world[i][125:375] = 1

#         return World(world, 250, 250)


# New world genreator to generate a maze
import random

class WorldGenerator:
    def __init__(self):
        return

    def new_world(self, grid):
        '''
        Generate random maze and randomly place robot in maze
        '''
        maze = np.ones((grid, grid), dtype=int)
        maze[1:-1, 1:-1] = 0  # make inner area of the maze empty

        # Recursive Backtracking algorithm to generate the maze
        stack = [(2, 2)]
        visited = set(stack)

        while stack:
            x, y = stack[-1]

            # find neighbors
            neighbors = [(x-2, y), (x+2, y), (x, y-2), (x, y+2)]
            neighbors = [(nx, ny) for nx, ny in neighbors if nx > 0 and ny > 0 and nx < grid-1 and ny < grid-1]
            neighbors = [(nx, ny) for nx, ny in neighbors if (nx, ny) not in visited]

            if neighbors:
                nx, ny = random.choice(neighbors)
                # knock down the wall between the current cell and the chosen neighbor
                if nx == x - 2:
                    maze[nx+1, y] = 0
                elif nx == x + 2:
                    maze[nx-1, y] = 0
                elif ny == y - 2:
                    maze[x, ny+1] = 0
                else:
                    maze[x, ny-1] = 0

                stack.append((nx, ny))
                visited.add((nx, ny))
            else:
                stack.pop()

        # randomly place robot in maze
        robot_x, robot_y = random.choice(np.argwhere(maze == 0))
        maze[robot_x, robot_y] = -1

        return World(maze, robot_x, robot_y)


World(maze, robot_x, robot_y)
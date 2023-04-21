import math
from collections import deque

'''
returns true if out of bound of map
'''
def out_of_bounds(x, y, xl, yl):
    return x < 0 or x >= xl or y < 0 or y >= yl

'''
map action from [0,1] to [-pi,pi]
'''
def action_to_rad(action):
    return action * math.pi

def rad_to_action(rad):
    return rad / math.pi

def index_to_rad(index):
    return (index - 0.5) * math.pi * 2
'''
Bresenham line algo from wikipedia
@param update_func: function that is called at every pixel,
                    takes in dx and dy so far
@param term_cond: terminating condition to stop algom
                  takes in dx and dy so far
@param x0: starting x position
@param y0: starting y position
@d: distance to travel
@heading: heading to travel

@return: total dx and dy from start point
'''
def bresenham_line(update_func, term_cond, x0, y0, d, heading):
    xs, ys = x0, y0
    x1 = int(x0 + d*math.cos(heading))
    y1 = int(y0 + d*math.sin(heading))
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        if term_cond(x0 - xs, y0 - ys):
            break
        update_func(x0 - xs, y0 - ys)
        if x0 == x1 and y0 == y1:
            break
        e2 = error*2
        if e2 >= dy:
            if x0 == x1:
                break
            error += dy
            x0 += sx
        if e2 <= dx:
            if y0 == y1:
                break
            error += dx
            y0 += sy
    return x0,y0

'''
Euclidean distance
'''
def euc_dist(x0, x1, y0, y1):
    return math.sqrt((x0-x1)**2+(y0-y1)**2)


def bfs_new(map_, start):
    neigh_func = lambda x,y: [
        (x-1,y-1), (x-1,y  ),(x-1,y+1),
        (x  ,y-1),           (x  ,y+1),
        (x+1,y-1), (x+1,y  ),(x+1,y+1)
        ]
    visited = set()
    queue = deque([(start[0], start[1])])
    while len(queue) != 0:
        curr = queue.popleft()
        neigh_lst = neigh_func(curr[0], curr[1])
        for a,b in neigh_lst:
            if not out_of_bounds(a, b, map_.shape[0], map_.shape[1]):
                if map_[a,b] == -1:
                    return ((curr[0], curr[1]))
                if map_[a,b] != 0 and (a,b) not in visited:
                    queue.append((a,b))
                    visited.add((a,b))
                
    return None
import math

'''
returns true if out of bound of map
'''
def out_of_bounds(x, y, xl, yl):
    return x < 0 and x >= xl and y < 0 and y >= yl

'''
map action from [0,1] to [-pi,pi]
'''
def action_to_rad(action):
    return (action - 0.5) * math.pi * 2

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
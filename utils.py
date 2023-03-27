import math

def out_of_bounds(x, y, size):
    return x < size and x >= 0 and y < size and y >= 0

# map action from [0,1] to [-pi,pi]
def action_to_rad(action):
    return (action - 0.5) * math.pi * 2
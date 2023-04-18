import numpy as np
import utils
import heapq

def a_star(map, start, end):
    neigh_func = lambda x,y: [
        (x-1,y-1), (x-1,y  ),(x-1,y+1),
        (x  ,y-1),           (x  ,y+1),
        (x+1,y-1), (x+1,y  ),(x+1,y+1)
        ]
    # preprocessing
    arr1 = []
    for i in range(map.shape[0]):
        arr2 = []
        for j in range(map.shape[1]):
            node = Node(i, j, utils.euc_dist(i, end[0], j, end[1]))
            arr2.append(node)
        arr1.append(arr2)
    nodes = np.array(arr1)
    # adding neighbours
    for i in range(nodes.shape[0]):
        for j in range(nodes.shape[1]):
            node = nodes[i,j]
            neigh_lst = neigh_func(i,j)
            for a,b in neigh_lst:
                if not utils.out_of_bounds(a, b, nodes.shape[0], nodes.shape[1]):
                    node.neighbour.append(nodes[a, b])
            # update node
            if map[i, j] != 1:
                node.obs = True
                # for no in node.neighbour:
                #     no.obs = True
    q = MyHeap(key=lambda x:x.find_cost())
    nodes[start[0], start[1]].g = 0
    q.push(nodes[start[0], start[1]])
    while not q.isEmpty():
        curr = q.pop()
        if (curr.x == end[0] and curr.y == end[1]):
            break
        for node in curr.neighbour:
            if node.obs or node.g <= curr.g + 1:
                continue
            node.parent = curr
            node.g = curr.g + 1
            q.push(node)
    curr = nodes[end[0], end[1]]
    if (curr.parent != None): # path found
        res = []
        while not (curr.x == start[0] and curr.y == start[1]):
            res.append((curr.x, curr.y))
            curr = curr.parent
        res.reverse()
        return res
    return None

class Node:
    def __init__(self, x, y, h):
        self.obs = False
        self.x = x
        self.y = y
        self.parent = None
        self.neighbour = []
        self.g = 9999999 # arbitrary large number
        self.h = h # heuristic cost, use euclidean

    def find_cost(self):
        return self.h + self.g 

class MyHeap(object):
    def __init__(self, initial=None, key=lambda x:x):
        self.key = key
        self.index = 0
        if initial:
            self._data = [(key(item), i, item) for i, item in enumerate(initial)]
            self.index = len(self._data)
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        heapq.heappush(self._data, (self.key(item), self.index, item))
        self.index += 1

    def pop(self):
        return heapq.heappop(self._data)[2]
    
    def isEmpty(self):
        return len(self._data) == 0
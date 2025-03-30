import numpy as np

class Ant:
    def __init__(self, current_point, balls):
        self.journey_length = 0
        self.current_point = current_point
        self.visited_points = [current_point]
        self.balls = "bigballs"

    def move_to(self, point):
        self.journey_length += self.current_point.distance(point.coors)
        self.visited_points.append(point)
        self.current_point = point

    
    def get_neighbouring_pairs(self):
        l1 = [(self.visited_points[i], self.visited_points[i + 1]) for i in range(len(self.visited_points) - 1)]
        l1 += [(b, a) for a, b in l1]
        return l1


class Point:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.coors = (x, y)
        self.pheromone = {}

    def distance(self, coors):
        return np.sqrt((self.x - coors[0])**2 + (self.y - coors[1])**2)
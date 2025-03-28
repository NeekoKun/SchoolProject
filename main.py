from utils import Point, Ant
import numpy as np
import logging
import random
import pygame

class Simulation:
    def __init__(self, size=(1920, 1080), points = 50, ants = 40, pheromone_weight = 10, pheromone_evaporation = 0.9, point_radius = 10, ant_radius = 5):
        pygame.init()
        self.screen = pygame.display.set_mode(size)

        self.background_color = pygame.Color("#F5DFBB")
        self.point_color = pygame.Color("#127475")
        self.pheromone_color = pygame.Color("#0E9594")
        self.ant_color = pygame.Color("#562C2C")

        self.point_radius = point_radius
        self.ant_radius = ant_radius

        self.size = self.width, self.height = size
        self.pheromone_weight = pheromone_weight
        self.pheromone_evaporation = pheromone_evaporation
        self.generate_points(points)
        self.generate_ants(ants)
    
    def run(self, time, gen_time, generations = 10):
        self.display()
    
        for _ in range(generations):
            ## Simulate a generation
            for _ in range(len(self.points) - 1):
                self.iter_simulation()
                self.display()
                pygame.time.wait(time)

            ## Lay the pheromone trails
            for ant in self.ants:
                for start, destination in ant.get_neighbouring_pairs():
                    self.points[start.id].pheromone[destination.id] += (len(self.points) * self.width * self.height) / (ant.journey_length) ** 2 #TODO: tune settings

            ## Evaporate pheromone
            for point in self.points:
                for pheromone in point.pheromone:
                    point.pheromone[pheromone] *= self.pheromone_evaporation

            ## Generate new ants
            self.generate_ants(len(self.ants))
            pygame.time.wait(gen_time)
    


    def iter_simulation(self):
        for ant in self.ants:
            targets = []
            weights = []

            for point in self.points:
                if point not in ant.visited_points:
                    targets.append(point)
                    weights.append((1 / ant.current_point.distance(point.coors)) + self.pheromone_weight * ant.current_point.pheromone[point.id])
            
            choice = np.random.choice(targets, p=np.array(weights) / np.sum(weights))
            ant.move_to(choice)


    def generate_ants(self, target_count):
        self.ants = []

        for i in range(target_count):
            ant = Ant(self.points[0])

            self.ants.append(ant)

    def generate_points(self, target_count):
        self.points = []

        for i in range(target_count):
            x = np.random.randint(0, self.size[0])
            y = np.random.randint(0, self.size[1])
            point = Point(i, x, y)

            self.points.append(point)

        for point in self.points:
            point.pheromone = {p.id: 0 for p in self.points if p != point}

    def display(self):
        self.screen.fill(self.background_color)

        for point in self.points:
            # Draw pheromone
            for pheromone in point.pheromone:
                pygame.draw.line(self.screen, self.pheromone_color, (point.x, point.y), (self.points[pheromone].x, self.points[pheromone].y), int(point.pheromone[pheromone]))
        
        pygame.draw.circle(self.screen, (255, 0, 0), (self.points[0].x, self.points[0].y), self.point_radius)
        for point in self.points[1:]:
            # Draw points
            pygame.draw.circle(self.screen, self.point_color, (point.x, point.y), self.point_radius)

        ## Display best ant
        best_ant = sorted(self.ants, key=lambda x: x.journey_length)[0]
        for i in range(len(best_ant.visited_points) - 1):
            pygame.draw.line(self.screen, self.ant_color, (best_ant.visited_points[i].x, best_ant.visited_points[i].y), (best_ant.visited_points[i+1].x, best_ant.visited_points[i+1].y), 5)
        
        """
        # Draw ants
        for ant in self.ants:
            pygame.draw.circle(self.screen, self.ant_color, (ant.current_point.x, ant.current_point.y), self.ant_radius)
        """
        pygame.display.flip()

sim = Simulation()

while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            sim.iter_simulation()
        if event.type == pygame.QUIT:
            pygame.quit()
            break
    
    sim.run(0, 0)
from utils import Ant
from scipy.spatial import KDTree
import numpy as np
import pygame
import logging
import os

class ColonySim:
    def __init__(self, ants, colony_location=None, food_locations=None, food_radius=None):
        self.frame_count = 0
        self.output_dir = "frames"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.map_size = (1000, 1000)
        
        self.colony_location = colony_location if colony_location else (500, 500)
        self.colony_radius = 10
        
        self.food_radius = food_radius if food_radius else 6
        self.food_locations = food_locations if food_locations else [(200, 200)]
        
        self.home_pheromones = {}
        self.food_pheromones = {}

        self.home_pheromones_tree = KDTree(np.empty((0, 2)))
        self.food_pheromones_tree = KDTree(np.empty((0, 2)))

        pygame.display.init()
        self.screen = pygame.display.set_mode(self.map_size)
        self.clock = pygame.time.Clock()

        self.ant_vision_radius = 50
        self.ants = [Ant(self.colony_location, vision_radius=self.ant_vision_radius) for _ in range(ants)]
    
        self.pheromone_depletion_function = lambda x: 10*(1 - 1/(1 + np.exp((40 - x)/10)))

    def update_pheromone_trees(self):
        home_pheromone_locations = np.array(list(self.home_pheromones.keys()))
        food_pheromone_locations = np.array(list(self.food_pheromones.keys()))

        if len(home_pheromone_locations) > 0:
            self.home_pheromones_tree = KDTree(home_pheromone_locations)
        else:
            self.home_pheromones_tree = KDTree(np.empty((0, 2)))

        if len(food_pheromone_locations) > 0:
            self.food_pheromones_tree = KDTree(food_pheromone_locations)
        else:
            self.food_pheromones_tree = KDTree(np.empty((0, 2)))

    def iterate_simulation(self):
        ## Move ants
        for ant in self.ants:
            if ant.has_food:
                if len(self.home_pheromones) == 0:
                    pheromones = []
                else:
                    pheromones_indices = self.home_pheromones_tree.query_ball_point(ant.location, self.ant_vision_radius)
                    pheromones = [((tuple(self.home_pheromones_tree.data[i])), self.home_pheromones[tuple(self.home_pheromones_tree.data[i])]) for i in pheromones_indices]
                ant.move(self.food_locations, self.food_radius, self.colony_location, self.colony_radius, self.map_size, pheromones)

            else:
                if len(self.food_pheromones) == 0:
                    pheromones = []
                else:
                    pheromones_indices = self.food_pheromones_tree.query_ball_point(ant.location, self.ant_vision_radius)
                    pheromones = [((tuple(self.food_pheromones_tree.data[i])), self.food_pheromones[tuple(self.food_pheromones_tree.data[i])]) for i in pheromones_indices]
                ant.move(self.food_locations, self.food_radius, self.colony_location, self.colony_radius, self.map_size, pheromones)

        ## Update pheromones
        for ant in self.ants:
            if ant.has_food:
                self.food_pheromones[ant.location] = self.pheromone_depletion_function(ant.steps)
            else:
                self.home_pheromones[ant.location] = self.pheromone_depletion_function(ant.steps)

        ## Dacay pheromones
        for home_pheromone in list(self.home_pheromones.keys()):
            self.home_pheromones[home_pheromone] *= 0.99
            if self.home_pheromones[home_pheromone] <= 0:
                del self.home_pheromones[home_pheromone]
        
        for food_pheromone in list(self.food_pheromones.keys()):
            self.food_pheromones[food_pheromone] *= 0.99
            if self.food_pheromones[food_pheromone] <= 0:
                del self.food_pheromones[food_pheromone]

        self.update_pheromone_trees()

    def display(self):
        self.screen.fill((7, 7, 9))
        pygame.draw.circle(self.screen, (0, 255, 0), self.colony_location, self.colony_radius)

        for food in self.food_locations:
            pygame.draw.circle(self.screen, (255, 0, 0), food, self.food_radius)
        
        
        for home_pheromone in self.home_pheromones:
            pygame.draw.circle(self.screen, (0, 0, 255), home_pheromone, self.home_pheromones[home_pheromone] / 3)
        for food_pheromone in self.food_pheromones:
            pygame.draw.circle(self.screen, (255, 0, 255), food_pheromone, self.food_pheromones[food_pheromone] / 3)
       

        for ant in self.ants:
            if ant.has_food:
                pygame.draw.circle(self.screen, (255, 255, 0), ant.location, 2)
            else:
                pygame.draw.circle(self.screen, (100, 100, 100), ant.location, 2)
        
        pygame.image.save(self.screen, f"{self.output_dir}/frame_{self.frame_count:05d}.png")
        self.frame_count += 1


    def run(self):
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.display()
            self.iterate_simulation()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
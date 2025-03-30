from utils import Ant
import numpy as np
import pygame
import random
import logging
import multiprocessing
import concurrent.futures
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


        pygame.display.init()
        self.screen = pygame.display.set_mode(self.map_size)
        self.clock = pygame.time.Clock()

        self.ants = [Ant(self.colony_location) for _ in range(ants)]

    def process_ant(self, ant):
        result = ant.move(self.food_locations, self.food_radius, self.colony_location, self.colony_radius, self.map_size, self.home_pheromones, self.food_pheromones)
        return ant, result
    
    def iterate_simulation(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = list(executor.map(self.process_ant, self.ants))

        for ant, result in results:
            if result == "Collected food":
                logging.info(f"Ant {ant} collected food at {ant.location}")
            elif result == "Deposited food":
                logging.info(f"Ant {ant} deposited food at {ant.location}")
            else:
                if ant.has_food:
                    self.food_pheromones[ant.location] = 10
                else:
                    self.home_pheromones[ant.location] = 10

        for home_pheromone in list(self.home_pheromones.keys()):
            self.home_pheromones[home_pheromone] -= 0.1
            if self.home_pheromones[home_pheromone] <= 0:
                del self.home_pheromones[home_pheromone]
        
        for food_pheromone in list(self.food_pheromones.keys()):
            self.food_pheromones[food_pheromone] -= 0.1
            if self.food_pheromones[food_pheromone] <= 0:
                del self.food_pheromones[food_pheromone]

    def display(self):
        self.screen.fill((7, 7, 9))
        pygame.draw.circle(self.screen, (0, 255, 0), self.colony_location, self.colony_radius)

        for food in self.food_locations:
            pygame.draw.circle(self.screen, (255, 0, 0), food, self.food_radius)
        
        """
        for home_pheromone in self.home_pheromones:
            pygame.draw.circle(self.screen, (0, 0, 255), home_pheromone, self.home_pheromones[home_pheromone] / 3)
        for food_pheromone in self.food_pheromones:
            pygame.draw.circle(self.screen, (255, 0, 255), food_pheromone, self.food_pheromones[food_pheromone] / 3)
    "   """

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
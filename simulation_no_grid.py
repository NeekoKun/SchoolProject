from utils import Ant
import numpy as np
import pygame
import random
import logging

class ColonySim:
    def __init__(self, ants, colony_location=None, food_locations=None, food_radius=None, food_amount=1000):
        self.map_size = (1000, 1000)
        self.colony_location = colony_location if colony_location else (500, 500)
        self.colony_radius = 10
        self.food_radius = food_radius if food_radius else 6
        self.home_pheromones = {}
        self.food_pheromones = {}

        food_locations = food_locations if food_locations else [(200, 200)]

        pygame.display.init()
        self.screen = pygame.display.set_mode(self.map_size)
        self.clock = pygame.time.Clock()

        for food_location in food_locations:
            self.food = [(food_location[0] + self.food_radius * np.cos(i * 31241), food_location[1] + self.food_radius * np.sin(i * 31241)) for i in range(food_amount)] 
    
        self.ants = [Ant(self.colony_location) for _ in range(ants)]

    def iterate_simulation(self):
        for ant in self.ants:
            result = ant.move(self.food, self.colony_location, self.map_size, self.home_pheromones, self.food_pheromones)
            if result == "Collected food":
                logging.info(f"Ant {ant} collected food at {ant.location}")
            elif result == "Deposited food":
                logging.info(f"Ant {ant} deposited food at {ant.location}")
            else:
                if ant.has_food:
                    self.food_pheromones[ant.location] = 10
                else:
                    self.home_pheromones[ant.location] = 10

            if ant.location == self.colony_location:
                ant.has_food = True
            elif ant.location in self.food:
                ant.has_food = False
                self.food.remove(ant.location)

        for home_pheromone in self.home_pheromones:
            self.home_pheromones[home_pheromone] -= 0.1
        
        for food_pheromone in self.food_pheromones:
            self.food_pheromones[food_pheromone] -= 0.1

    def display(self):
        self.screen.fill((7, 7, 9))
        pygame.draw.circle(self.screen, (0, 255, 0), self.colony_location, self.colony_radius)

        for food in self.food:
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
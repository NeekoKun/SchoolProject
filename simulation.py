
import numpy as np
import random
import pygame

class ColonySim:
    def __init__(self):
        self.SIZE = self.WIDTH, self.HEIGHT = 1000, 1000

        pygame.display.init()
        self.screen = pygame.display.set_mode(self.SIZE)
        self.background_grid = np.zeros(self.SIZE)

    def generate_cave(self):
        # Generate random map
        for x, row in enumerate(self.background_grid):
            for y, _ in enumerate(row):
                if random.randint(0, 100) < 45:
                    self.background_grid[y][x] = 1

        for _ in range(5):
            buffer = self.background_grid.copy()

            for y, row in enumerate(self.background_grid):
                if (y == 0) or (y == len(self.SIZE[1]) - 1):
                    continue
                for x, cell in enumerate(row):
                    if (x == 0) or (x == len(self.SIZE[0]) - 1):
                        continue
                    # Count walls around cell
                    walls = 0

                    if self.background_grid[y+1][x+1] == 1:
                        walls += 1
                    if self.background_grid[y+1][x] == 1:
                        walls += 1
                    if self.background_grid[y+1][x-1] == 1:
                        walls += 1
                    if self.background_grid[y][x+1] == 1:
                        walls += 1
                    if self.background_grid[y][x-1] == 1:
                        walls += 1
                    if self.background_grid[y-1][x+1] == 1:
                        walls += 1
                    if self.background_grid[y-1][x] == 1:
                        walls += 1
                    if self.background_grid[y-1][x-1] == 1:
                        walls += 1
                
                    if walls >= 5:
                        buffer[y][x] = 1
                    else:
                        buffer[y][x] = 0
            
            self.background_grid = buffer.copy()


    def iter_simulation(self):
        pass

    def display(self):
        for y, row in enumerate(self.background_grid):
            for x, cell in enumerate(row):
                self.screen.set_at((x, y), (cell*255, cell*255, cell*255))


    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            self.iter_simulation()
            self.display()

sim = ColonySim()
sim.generate_cave()
sim.run()


import scipy.ndimage as ndimage
import numpy as np
import logging
import random
import pygame
import time
import json

logging.basicConfig(level=logging.DEBUG)

class ColonySim:
    def __init__(self, seed=None):
        self.SIZE = self.WIDTH, self.HEIGHT = 100, 100
        self.CELL_SIZE = self.CELL_WIDTH, self.CELL_HEIGHT = 10, 10

        pygame.display.init()
        self.screen = pygame.display.set_mode((self.WIDTH * self.CELL_WIDTH, self.HEIGHT*self.CELL_HEIGHT))
        
        self.colors = json.load(open("colors.json"))

        if seed is None:
            seed = random.randint(0, 1000000)
        random.seed(seed)

        logging.debug(f"Seed: {seed}")
        self.walls_grid = np.zeros(self.SIZE)

    def convolute(self, m, kernel, exterior=1):
        matrix = m.copy()
        kernel_height, kernel_width = kernel.shape
        matrix_height, matrix_width = matrix.shape
        output = np.zeros((matrix_height, matrix_width))

        # Check if kernel is odd
        if kernel_height % 2 == 0 or kernel_width % 2 == 0:
            raise ValueError("Kernel must have odd dimensions")
        
        # Actual convolution
        padded_matrix = np.pad(matrix, ((kernel_height // 2, kernel_height // 2), (kernel_width // 2, kernel_width // 2)), constant_values=exterior)
        for i in range(matrix_height):
            for j in range(matrix_width):
                region = padded_matrix[i:i + kernel_height, j:j + kernel_width]
                output[i, j] = np.sum(region * kernel)
        
        return output

    def cellular_automata(self, m, rules, iterations, display=False):
        matrix = m.copy()
        logging.debug(f"Applying cellular automata with rules: {rules} for {iterations} iterations")
        # Apply rules to the matrix
        for i in range(iterations):
            logging.debug(f"Iteration {i+1}/{iterations}")
            buffer = np.zeros(self.SIZE)
            convolutions = {}

            for rule in rules:
                convolutions[rule] = self.convolute(matrix, np.ones((2*rule+1, 2*rule+1)))

            for y in range(matrix.shape[0]):
                for x in range(matrix.shape[1]):
                    # Get rules
                    
                    for rule in rules:
                        # Apply rules
                        if rules[rule] < 0:
                            # Invert the rule
                            if convolutions[rule][y][x] <= -rules[rule]:
                                buffer[y][x] = 1
                        else:
                            # Normal rule
                            if convolutions[rule][y][x] >= rules[rule]:
                                buffer[y][x] = 1

            # Update the matrix
            matrix = buffer.copy()

            if display:
                self.display([(matrix, self.colors["walls"])])
                pygame.time.wait(500)

        return matrix

    def generate_cave(self, display=False, display_steps=10):
        # Generate random map
        for x, row in enumerate(self.walls_grid):
            for y, _ in enumerate(row):
                if random.randint(0, 100) < 45:
                    self.walls_grid[y][x] = 1
                else:
                    self.walls_grid[y][x] = 0

        if display:
            self.display([(self.walls_grid, (255, 255, 255))])
            pygame.time.wait(500)

        self.walls_grid = self.cellular_automata(self.walls_grid, {1: 5, 2:-2}, 4, display=display)        
        self.walls_grid = self.cellular_automata(self.walls_grid, {1: 5}, 4, display=display)

        start_flood_time = time.time()

        ## Flood fill to prevent isolated areas
        visited = np.zeros(self.SIZE)
        stack = []

        # Generate starter point to flood fill
        stack.append((0, 0))
        while self.walls_grid[stack[-1][0]][stack[-1][1]] == 1:
            stack.pop()
            stack.append((random.randint(0, self.WIDTH-1), random.randint(0, self.HEIGHT-1)))

        cycle = 0
        # Flood fill following stack order
        while len(stack) > 0:
            # Pop the last element
            (x, y) = stack.pop()

            # Skip if already visited
            if visited[y][x] == 1:
                continue

            visited[y][x] = 1

            # Check if the cell is a wall
            if x > 0 and visited[y][x-1] == 0 and self.walls_grid[y][x-1] == 0:
                stack.append((x-1, y))
            if x < self.WIDTH-1 and visited[y][x+1] == 0 and self.walls_grid[y][x+1] == 0:
                stack.append((x+1, y))
            if y > 0 and visited[y-1][x] == 0 and self.walls_grid[y-1][x] == 0:
                stack.append((x, y-1))
            if y < self.HEIGHT-1 and visited[y+1][x] == 0 and self.walls_grid[y+1][x] == 0:
                stack.append((x, y+1))

            # Display for debugging purposes
            if display and cycle % display_steps == 0:
                # Display the background grid and the visited cells
                self.display([(self.walls_grid, self.colors["walls"]), (visited, (255, 0, 0))], squares = [(x, y)])

            cycle += 1

        walls = self.walls_grid.sum()
        total = self.WIDTH * self.HEIGHT
        logging.debug(f"Free area: {100 - (walls / total) * 100}%")

        end_flood_time = time.time()

        logging.debug(f"Flood fill took {end_flood_time - start_flood_time} seconds")

        # Remove isolated areas
        for y in range(1, self.HEIGHT-1):
            for x in range(1, self.WIDTH-1):
                if visited[y][x] == 0:
                    self.walls_grid[y][x] = 1

        # Get current free area percentage
        walls = self.walls_grid.sum()
        total = self.WIDTH * self.HEIGHT
        logging.debug(f"Free area: {100 - (walls / total) * 100}%")

        if 100 - (walls / total) * 100 < 45:
            logging.debug("Cave generation failed, regenerating")
            self.generate_cave(display=display, display_steps=display_steps)

        return self.walls_grid

    def generate_colony(self, location=None, radius=5, display=False):
        if location:
            self.colony_location = location
        else:
            ## Generate distance map from the walls of the cave
            distance_map = ndimage.distance_transform_edt(self.walls_grid == 0)

            # Normalize the distance map to the range [0, 1]
            distance_map = distance_map / np.max(distance_map)
            
            # Display the distance map
            if display:
                self.display([(distance_map, (0, 255, 0)), (self.walls_grid, self.colors["walls"])]) # TODO: Remove in production
                pygame.time.wait(500)

            # Find the maximum value in the distance map
            self.colony_location = [0, 0]
            self.colony_location[1], self.colony_location[0] = np.unravel_index(np.argmax(distance_map), distance_map.shape)
        
        # Display maximum distance point
        if display:
            self.display([(distance_map, (0, 255, 0))], squares=[self.colony_location])
            pygame.time.wait(500)
        
        ## Generate colony
        self.colony_grid = np.zeros(self.SIZE)
        for y in range(self.colony_location[1] - radius, self.colony_location[1] + radius):
            for x in range(self.colony_location[0] - radius, self.colony_location[0] + radius):
                if (x - self.colony_location[0])**2 + (y - self.colony_location[1])**2 < radius**2:
                    if self.walls_grid[y][x] == 0:
                        self.colony_grid[y][x] = 1
        # Display colony
        if display:
            self.display([(self.colony_grid, self.colors["colony"]), (self.walls_grid, self.colors["walls"])], squares=[self.colony_location])
            pygame.time.wait(500)

    def generate_food_source(self, amount, location=None, radius=5, display=False, counter=0):
        # Check if there already is a food source
        if not hasattr(self, "food_grid"):
            self.food_grid = np.zeros(self.SIZE)

        if location:
            self.food_location = location
        else:
            # Generate distance map from colony
            if hasattr(self, "colony_grid"):
                colony_distance_map = ndimage.distance_transform_edt(self.colony_grid == 0)
                # Normalize the distance map to the range [0, 1]
                colony_distance_map = colony_distance_map / np.max(colony_distance_map)
            else:
                colony_distance_map = np.ones(self.SIZE)
            # Display the distance map
            if display:
                self.display([(colony_distance_map, (0, 255, 0)), (self.walls_grid, self.colors["walls"])])
                pygame.time.wait(500)

            # Generate distance map from the walls of the cave and other food sources
            walls_and_food = np.maximum(self.walls_grid, self.food_grid)
            walls_distance_map = ndimage.distance_transform_edt(walls_and_food == 0)
            # Normalize the distance map to the range [0, 1]
            walls_distance_map = walls_distance_map / np.max(walls_distance_map)
            # Display the distance map
            if display:
                self.display([(walls_distance_map, (0, 255, 0)), (self.walls_grid, self.colors["walls"])])
                pygame.time.wait(500)

            # Combine the distance maps
            distance_map = np.where(colony_distance_map > 0.5, walls_distance_map, 0)
            if display:
                self.display([(distance_map, (0, 255, 0)), (self.walls_grid, self.colors["walls"])])
                pygame.time.wait(500)

            # Find the maximum value in the distance map
            self.food_location = [0, 0]
            self.food_location[1], self.food_location[0] = np.unravel_index(np.argmax(distance_map), distance_map.shape)

            # Display maximum distance point
            if display:
                self.display([(distance_map, (0, 255, 0))], squares=[self.food_location])
                pygame.time.wait(500)

        # Display food location
        if display:
            self.display([(self.colony_grid, self.colors["colony"]), (self.walls_grid, self.colors["walls"])], squares=[self.food_location])
            pygame.time.wait(500)
        
        ## Generate food source
        for y in range(self.food_location[1] - radius, self.food_location[1] + radius):
            for x in range(self.food_location[0] - radius, self.food_location[0] + radius):
                if (x - self.food_location[0])**2 + (y - self.food_location[1])**2 < radius**2:
                    if self.walls_grid[y][x] == 0:
                        self.food_grid[y][x] = 1

        counter += 1

        # Display food source
        if display:
            self.display([(self.food_grid, self.colors["food"]), (self.colony_grid, self.colors["colony"]), (self.walls_grid, self.colors["walls"])], squares=[self.food_location])
            pygame.time.wait(500)

        if counter < amount:
            # Generate food source again
            self.generate_food_source(amount, location=None, radius=radius, display=display, counter=counter)


    def iter_simulation(self):
        pass

    def display(self, matrix_list=None, squares=[], background=None):
        if background:
            self.screen.fill(background)
        else:
            self.screen.fill(self.colors["background"])
        ## Generate matrix to display
        if matrix_list is None:
            matrix_list = [(self.walls_grid, (255, 255, 255))]
        
        for matrix, color in matrix_list:
            for y, row in enumerate(matrix):
                for x, cell in enumerate(row):
                    if cell == 0:
                        continue
                    rect = pygame.Rect(x * self.CELL_WIDTH, y * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
                    cell_color = (cell * color[0], cell * color[1], cell * color[2])
                    pygame.draw.rect(self.screen, cell_color, rect)
        
        for square in squares:
            x, y = square
            rect = pygame.Rect(x * self.CELL_WIDTH, y * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
            pygame.draw.rect(self.screen, (0, 0, 255), rect)
        
        pygame.display.flip()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logging.debug("Quitting")
                    pygame.quit()
                    exit()

            self.iter_simulation()
            
            self.display([
                (self.walls_grid, self.colors["walls"]),
                (self.colony_grid, self.colors["colony"]),
                (self.food_grid, self.colors["food"])
            ])

sim = ColonySim(297947)
logging.debug("Starting cave Generation")
sim.generate_cave(display=True, display_steps=10)
sim.generate_colony(display=True)
sim.generate_food_source(2, display=True)
logging.debug("Cave Generation complete")
logging.debug("Starting simulation")
sim.run()


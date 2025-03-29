import scipy.ndimage as ndimage
import numpy as np
import logging
import random
import pygame
import time

logging.basicConfig(level=logging.DEBUG)

class ColonySim:
    def __init__(self, seed=None):
        self.SIZE = self.WIDTH, self.HEIGHT = 100, 100
        self.CELL_SIZE = self.CELL_WIDTH, self.CELL_HEIGHT = 10, 10

        pygame.display.init()
        self.screen = pygame.display.set_mode((self.WIDTH * self.CELL_WIDTH, self.HEIGHT*self.CELL_HEIGHT))
        
        random.seed(seed) if seed else random.seed()
        logging.debug(f"Seed: {seed}")
        self.background_grid = np.zeros(self.SIZE)

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
                            logging.info(f"Applying rule {rule} [{convolutions[rule][y][x]}] with value <= {-rules[rule]}")
                            if convolutions[rule][y][x] <= -rules[rule]:
                                buffer[y][x] = 1
                        else:
                            # Normal rule
                            logging.info(f"Applying rule {rule} [{convolutions[rule][y][x]}] with value >= {rules[rule]}")
                            if convolutions[rule][y][x] >= rules[rule]:
                                buffer[y][x] = 1

            # Update the matrix
            matrix = buffer.copy()

            if display:
                self.display([(matrix, (255, 255, 255))])
                pygame.time.wait(500)

        return matrix


    def generate_cave(self, display=False, display_steps=10):
        # Generate random map
        for x, row in enumerate(self.background_grid):
            for y, _ in enumerate(row):
                if random.randint(0, 100) < 45:
                    self.background_grid[y][x] = 1

        if display:
            self.display([(self.background_grid, (255, 255, 255))])
            pygame.time.wait(500)

        self.background_grid = self.cellular_automata(self.background_grid, {1: 5, 2:-2}, 4, display=display)        
        self.background_grid = self.cellular_automata(self.background_grid, {1: 5}, 3, display=display)

        start_flood_time = time.time()

        ## Flood fill to prevent isolated areas
        visited = np.zeros(self.SIZE)
        stack = []

        # Generate starter point to flood fill
        stack.append((0, 0))
        while self.background_grid[stack[-1][0]][stack[-1][1]] == 1:
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
            if x > 0 and visited[y][x-1] == 0 and self.background_grid[y][x-1] == 0:
                stack.append((x-1, y))
            if x < self.WIDTH-1 and visited[y][x+1] == 0 and self.background_grid[y][x+1] == 0:
                stack.append((x+1, y))
            if y > 0 and visited[y-1][x] == 0 and self.background_grid[y-1][x] == 0:
                stack.append((x, y-1))
            if y < self.HEIGHT-1 and visited[y+1][x] == 0 and self.background_grid[y+1][x] == 0:
                stack.append((x, y+1))

            # Display for debugging purposes
            if display and cycle % display_steps == 0:
                # Display the background grid and the visited cells
                self.display([(self.background_grid, (255, 255, 255)), (visited, (255, 0, 0))], squares = [(x, y)])

            cycle += 1

        walls = self.background_grid.sum()
        total = self.WIDTH * self.HEIGHT
        logging.debug(f"Free area: {100 - (walls / total) * 100}%")

        if 100 - (walls / total) * 100 < 40:
            logging.debug("Filled area below 45%, retrying")
        else:
            return False

        end_flood_time = time.time()

        logging.debug(f"Flood fill took {end_flood_time - start_flood_time} seconds")

        # Remove isolated areas
        for y in range(1, self.HEIGHT-1):
            for x in range(1, self.WIDTH-1):
                if visited[y][x] == 0:
                    self.background_grid[y][x] = 1

        # Get current free area percentage

        return self.background_grid

    def generate_food_source(self):
        ## Generate distance map from the walls of the cave

        # Generate gaussian kernel
        kernel_size = 11
        kernel_deviation = 3
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, kernel_size//2] = 1
        kernel = ndimage.gaussian_filter(kernel, kernel_deviation)

        distance_map = self.convolute(self.background_grid, kernel)
        while True:
            self.display([(distance_map, (0, 255, 0)), (self.background_grid, (255, 255, 255))])
        

    def iter_simulation(self):
        pass

    def display(self, matrix_list=None, squares=[]):
        self.screen.fill((0, 0, 0))
        ## Generate matrix to display
        if matrix_list is None:
            matrix_list = [(self.background_grid, (255, 255, 255))]
        
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
            self.display()

sim = ColonySim(seed=2)
logging.debug("Starting cave Generation")
sim.generate_cave(display=True, display_steps=1)
#sim.generate_food_source()
#sim.generate_colony()
logging.debug("Cave Generation complete")
logging.debug("Starting simulation")
sim.run()


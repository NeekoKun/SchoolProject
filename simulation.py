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

    def convolute(self, matrix, kernel):
        kernel_height, kernel_width = kernel.shape
        matrix_height, matrix_width = matrix.shape
        output = np.zeros((matrix_height, matrix_width))

        # Check if kernel is odd
        if kernel_height % 2 == 0 or kernel_width % 2 == 0:
            raise ValueError("Kernel must have odd dimensions")
        
        for i in range(kernel_height // 2, matrix_height - kernel_height // 2):
            for j in range(kernel_width // 2, matrix_width - kernel_width // 2):
                region = matrix[i - kernel_height // 2:i + kernel_height // 2 + 1, j - kernel_width // 2:j + kernel_width // 2 + 1]
                output[i, j] = np.sum(region * kernel)
        
        return output

    def generate_cave(self):
        # Generate random map
        for x, row in enumerate(self.background_grid):
            for y, _ in enumerate(row):
                if random.randint(0, 100) < 45 or x == 0 or x == self.WIDTH-1 or y == 0 or y == self.HEIGHT-1:
                    self.background_grid[y][x] = 1

        # Apply cellular automata
        for _ in range(4):
            ## Rules:
            # 1. If a cell has 5 or more walls around it, it stays a wall
            # 2. If a cell has 2 or less walls around its 2 square neightbourhood, it becomes a wall

            buffer = np.ones(self.SIZE)

            # Get nieghtbour values through matrix convolution
            n1 = self.convolute(self.background_grid, np.ones((3, 3)))
            n2 = self.convolute(self.background_grid, np.ones((5, 5)))

            # Apply rules
            for y in range(1, self.HEIGHT-1):
                for x in range(1, self.WIDTH-1):
                    # Count walls around cell
                    walls1 = n1[y][x]
                    walls2 = n2[y][x]

                    if walls1 >= 5 or walls2 <= 2:
                        buffer[y][x] = 1
                    else:
                        buffer[y][x] = 0
            
            ## Update the background grid
            self.background_grid = buffer.copy()

            # Display for debugging purposes
            self.display()
            pygame.time.wait(500)

        for _ in range(3):
            ## Rules:
            # 1. If a cell has 5 or more walls around it, it stays a wall

            buffer = np.ones(self.SIZE)

            # Get nieghtbour values through matrix convolution
            n1 = self.convolute(self.background_grid, np.array([[0.5, 1, 0.5], [1, 1, 1], [0.5, 1, 0.5]]))

            # Apply rules
            for y in range(1, self.HEIGHT-1):
                for x in range(1, self.WIDTH-1):
                    # Count walls around cell
                    walls1 = n1[y][x]
                    walls2 = n2[y][x]

                    if walls1 >= 4:
                        buffer[y][x] = 1
                    else:
                        buffer[y][x] = 0
            
            ## Update the background grid
            self.background_grid = buffer.copy()

            # Display for debugging purposes
            self.display()
            pygame.time.wait(500)


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
            (x, y) = stack.pop(0)

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
            if cycle % 10 == 0:
                # Display the background grid and the visited cells
                self.display([(self.background_grid, (255, 255, 255)), (visited, (255, 0, 0))], squares = [(x, y)])

            cycle += 1


        end_flood_time = time.time()
        logging.debug(f"Flood fill took {end_flood_time - start_flood_time} seconds")

        # Remove isolated areas
        for y in range(1, self.HEIGHT-1):
            for x in range(1, self.WIDTH-1):
                if visited[y][x] == 0:
                    self.background_grid[y][x] = 1

        # Get current free area percentage
        walls = self.background_grid.sum()
        total = self.WIDTH * self.HEIGHT
        logging.debug(f"Free area: {100 - (walls / total) * 100}%")

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
sim.generate_cave()
logging.debug("Cave Generation complete")
logging.debug("Starting simulation")
sim.run()


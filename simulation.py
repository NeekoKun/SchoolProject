from concurrent.futures import ThreadPoolExecutor
import scipy.ndimage as ndimage
from utils import Ant
import numpy as np
import logging
import random
import pygame
import time
import json

logging.basicConfig(level=logging.DEBUG)

class ColonySim:
    def __init__(self, seed=None):
        """
        Initializes the simulation environment.
        Args:
            seed (int, optional): A seed value for random number generation. If not provided, 
                                  a random seed will be generated.
        Attributes:
            SIZE (tuple): The size of the simulation grid (WIDTH, HEIGHT).
            WIDTH (int): The width of the simulation grid in cells.
            HEIGHT (int): The height of the simulation grid in cells.
            CELL_SIZE (tuple): The size of each cell in pixels (CELL_WIDTH, CELL_HEIGHT).
            CELL_WIDTH (int): The width of each cell in pixels.
            CELL_HEIGHT (int): The height of each cell in pixels.
            screen (pygame.Surface): The Pygame display surface for rendering the simulation.
            colors (dict): A dictionary of colors loaded from a JSON file.
            walls_grid (numpy.ndarray): A 2D array representing the grid of walls in the simulation.
        """
        
        self.SIZE = self.WIDTH, self.HEIGHT = 100, 100
        self.CELL_SIZE = self.CELL_WIDTH, self.CELL_HEIGHT = 10, 10

        pygame.display.init()
        self.screen = pygame.display.set_mode((self.WIDTH * self.CELL_WIDTH, self.HEIGHT*self.CELL_HEIGHT))
        
        self.colors = json.load(open("colors.json"))
        self.colony_food = 0

        if seed is None:
            seed = random.randint(0, 1000000)
        random.seed(seed)

        logging.debug(f"Seed: {seed}")
        self.walls_grid = np.zeros(self.SIZE)

    def convolute(self, m: list[list], kernel: list[list], exterior=1):
        """
        Perform a 2D convolution operation on a matrix using a given kernel.
        Parameters:
            m (numpy.ndarray): The input matrix to be convolved.
            kernel (numpy.ndarray): The convolution kernel, which must have odd dimensions.
            exterior (int, optional): The value to use for padding the matrix edges. Defaults to 1.
        Returns:
            numpy.ndarray: The resulting matrix after applying the convolution.
        Raises:
            ValueError: If the kernel does not have odd dimensions.
        Notes:
            - The input matrix `m` is not modified; a copy is used for the operation.
            - The kernel is applied to the matrix with zero-padding (or the specified `exterior` value) 
              to handle edge cases.
        """

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
        """
        Applies a cellular automaton to a given matrix based on specified rules and iterations.
        Args:
            m (numpy.ndarray): The input matrix to which the cellular automaton will be applied.
            rules (dict): A dictionary where keys represent the rule size (radius) and values 
                          represent the threshold for activation. Positive values indicate 
                          activation when the sum of neighbors is greater than or equal to the 
                          threshold, while negative values indicate activation when the sum of 
                          neighbors is less than or equal to the absolute value of the threshold.
            iterations (int): The number of iterations to apply the cellular automaton.
            display (bool, optional): If True, displays the matrix after each iteration using 
                                      the `self.display` method. Defaults to False.
        Returns:
            numpy.ndarray: The resulting matrix after applying the cellular automaton for the 
                           specified number of iterations.
        Notes:
            - The method uses a convolution operation to calculate the sum of neighbors for 
              each cell based on the rule size.
            - The `self.display` method is used for visualization if `display` is set to True.
            - The `self.colors["walls"]` is used for coloring during visualization.
        """

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
        """
        Generates a cave-like structure using a combination of random initialization, 
        cellular automata, and flood fill to ensure connectivity.
        Args:
            display (bool, optional): If True, displays the cave generation process 
                visually for debugging purposes. Defaults to False.
            display_steps (int, optional): The number of steps between visual updates 
                when `display` is True. Defaults to 10.
        Returns:
            numpy.ndarray: A 2D grid representing the generated cave, where 1 indicates 
            a wall and 0 indicates free space.
        Notes:
            - The cave is initialized with a random distribution of walls and free spaces.
            - Cellular automata rules are applied to refine the cave structure.
            - A flood fill algorithm ensures there are no isolated areas, and all free 
              spaces are connected.
            - If the resulting cave has less than 45% free area, the generation process 
              is repeated recursively.
            - Debugging information, such as free area percentage and flood fill duration, 
              is logged.
        """

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

        #self.walls_grid = self.cellular_automata(self.walls_grid, {1: 5, 2:-2}, 4, display=display)        
        self.walls_grid = self.cellular_automata(self.walls_grid, {1: 5}, 5, display=display)

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
        """
        Generates a colony within the simulation environment.
        This function determines the location of the colony either based on a given
        location or by finding the point farthest from the walls of the cave. It then
        creates a circular colony grid centered at the chosen location with a specified
        radius. Optionally, the process can be visualized.
        Args:
            location (list or tuple, optional): The (x, y) coordinates of the colony's 
                location. If not provided, the location is determined automatically.
            radius (int, optional): The radius of the colony. Defaults to 5.
            display (bool, optional): Whether to display the process of colony generation.
                Defaults to False.
        Side Effects:
            - Updates `self.colony_location` with the chosen colony location.
            - Updates `self.colony_grid` with the generated colony grid.
        Notes:
            - The function uses a distance transform to find the point farthest from
              the walls if no location is provided.
            - The visualization is intended for debugging and should be removed in
              production.
        """
        
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
            # Mask out points within the radius distance from the border
            border_mask = np.zeros_like(distance_map, dtype=bool)
            border_mask[:radius, :] = True
            border_mask[-radius:, :] = True
            border_mask[:, :radius] = True
            border_mask[:, -radius:] = True

            # Apply the mask to the distance map
            masked_distance_map = np.where(border_mask, 0, distance_map)

            # Find the maximum value in the masked distance map
            max_distance = np.argmax(masked_distance_map)

            self.colony_location[1], self.colony_location[0] = np.unravel_index(max_distance, distance_map.shape)

            logging.debug(f"Colony location: {self.colony_location}")

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
        """
        Generates food sources in the simulation environment.
        This method creates food sources within the simulation grid. It ensures that
        food sources are placed at appropriate locations based on distance maps and
        avoids overlapping with walls or existing food sources.
        Args:
            amount (int): The total number of food sources to generate.
            location (tuple, optional): A specific (x, y) coordinate for the food source.
                If None, the location is determined based on distance maps. Defaults to None.
            radius (int, optional): The radius of the food source area. Defaults to 5.
            display (bool, optional): Whether to visually display the process of generating
                food sources. Defaults to False.
            counter (int, optional): Internal counter to track the number of food sources
                generated. Defaults to 0.
        Behavior:
            - If `location` is provided, the food source is generated at the specified
              location.
            - If `location` is not provided, distance maps are used to determine an
              optimal location for the food source.
            - The method ensures that food sources do not overlap with walls or other
              obstacles.
            - If `display` is True, intermediate steps and the final food source
              placement are visually displayed.
        Notes:
            - This method is recursive and will continue generating food sources until
              the specified `amount` is reached.
            - The `counter` argument is used internally to track the recursion depth
              and should not be manually set when calling the method.
        Raises:
            ValueError: If the specified `amount` is less than or equal to zero.
        """

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

            # Mask out points within the radius distance from the border
            border_mask = np.zeros_like(distance_map, dtype=bool)
            border_mask[:radius, :] = True
            border_mask[-radius:, :] = True
            border_mask[:, :radius] = True
            border_mask[:, -radius:] = True

            # Apply the mask to the distance map
            distance_map[border_mask] = 0

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

    def scale_grid(self, scale: list[int], display=False):
        if hasattr(self, "walls_grid"):
            self.walls_grid = np.kron(self.walls_grid, np.ones((scale, scale)))

        if hasattr(self, "colony_grid"):
            self.colony_grid = np.kron(self.colony_grid, np.ones((scale, scale)))
            self.colony_location = [self.colony_location[0] * scale + self.CELL_WIDTH//2, self.colony_location[1] * scale + self.CELL_HEIGHT//2]

        if hasattr(self, "food_grid"):
            self.food_grid = np.kron(self.food_grid, np.ones((scale, scale)))
        
        self.WIDTH = self.WIDTH * scale
        self.HEIGHT = self.HEIGHT * scale
        self.CELL_WIDTH = self.CELL_WIDTH // scale
        self.CELL_HEIGHT = self.CELL_HEIGHT // scale
        self.SIZE = self.WIDTH, self.HEIGHT
        self.screen = pygame.display.set_mode((self.WIDTH * self.CELL_WIDTH, self.HEIGHT*self.CELL_HEIGHT))
        
        if display:
            self.display([
                (self.walls_grid, self.colors["walls"]),
                (self.colony_grid, self.colors["colony"]),
                (self.food_grid, self.colors["food"])
            ])
            pygame.time.wait(500)

    def generate_ants(self, amount: int):
        """
        Generates a specified number of ants within the simulation environment.
        Args:
            amount (int): The number of ants to generate.
        Behavior:
            - Initializes an empty list of ants.
            - For each ant, creates an instance of the Ant class and appends it to the list.
        - The ants are initialized with the colony location.
        - The colony location is determined by the `self.colony_location` attribute.
        - The ants are stored in the `self.ants` attribute.
        Notes:
            - This method assumes that the `Ant` class is defined in the `utils` module.
            - The `self.colony_location` attribute should be set before calling this method.
        Raises:
            ValueError: If the `amount` is less than or equal to zero.
        Example:
            sim = ColonySim()
            sim.generate_colony()
            sim.generate_ants(10)
        """
        if amount <= 0:
            raise ValueError("Amount must be greater than 0")

        self.search_pheromone_grid = np.zeros(self.SIZE)
        self.food_pheromone_grid = np.zeros(self.SIZE)

        self.ants = []
        for i in range(amount):
            ant = Ant(self.colony_location)
            self.ants.append(ant)
    
    def iter_simulation(self):
        if not hasattr(self, "ants"):
            raise ValueError("Ants not generated yet")

        def process_ant(ant):
            # Move the ant
            result = ant.move(self.walls_grid, self.colony_grid, self.food_grid)
            # If the ant found food, get food, update status and turn around
            if result == "food" and not ant.has_food:
                ant.has_food = True
                ant.rotation += np.pi

            # If the ant is at the colony, drop food and turn around
            if result == "colony" and ant.has_food:
                ant.has_food = False
                ant.rotation += np.pi
                self.colony_food += 1

            if ant.has_food:
                # Update food pheromone grid
                self.food_pheromone_grid[ant.location[1]][ant.location[0]] = 1
                ant.random_rotate(self.search_pheromone_grid)
            else:
                # Update search pheromone grid
                self.search_pheromone_grid[ant.location[1]][ant.location[0]] = 1
                ant.random_rotate(self.food_pheromone_grid)

        with ThreadPoolExecutor() as executor:
            executor.map(process_ant, self.ants)

        # Decay pheromones
        self.search_pheromone_grid *= 0.9
        self.food_pheromone_grid *= 0.9
    
    """"
    def iter_simulation(self):
        if not hasattr(self, "ants"):
            raise ValueError("Ants not generated yet")

        for ant in self.ants:
            # Move the ant
            result = ant.move(self.walls_grid, self.colony_grid, self.food_grid)
            # If the ant found food, get food, update status and turn around
            if result == "food" and ant.has_food == False:
                ant.has_food = True
                ant.rotation += np.pi
            
            # If the ant is at the colony, drop food and turn around
            if result == "colony" and ant.has_food == True:
                ant.has_food = False
                ant.rotation += np.pi
                self.colony_food += 1
            
            if ant.has_food == True:
                # Update food pheromone grid
                self.food_pheromone_grid[ant.location[1]][ant.location[0]] = 1
                ant.random_rotate(self.search_pheromone_grid)        
            else:
                # Update search pheromone grid
                self.search_pheromone_grid[ant.location[1]][ant.location[0]] = 1
                ant.random_rotate(self.food_pheromone_grid)

        ## Decay pheromones
        self.search_pheromone_grid *= 0.9
        self.food_pheromone_grid *= 0.9
    """

    def display(self, matrix_list=None, squares=[], background=None, ants=True):
        """
        Renders a graphical representation of a grid-based simulation on the screen.
        Args:
            matrix_list (list of tuples, optional): A list of tuples where each tuple contains a 2D matrix 
                (list of lists) and a color tuple (R, G, B). Each matrix represents a grid to be displayed, 
                and the color tuple determines the base color for the cells in the matrix. Defaults to 
                [(self.walls_grid, (255, 255, 255))] if not provided.
            squares (list of tuples, optional): A list of (x, y) coordinates representing specific cells 
                to be highlighted in blue. Defaults to an empty list.
            background (tuple, optional): An RGB color tuple (R, G, B) to fill the screen background. 
                If not provided, the default background color from `self.colors["background"]` is used.
        Behavior:
            - Fills the screen with the specified or default background color.
            - Iterates through the provided matrices in `matrix_list` and renders each cell with its 
              corresponding color.
            - Highlights specific cells in `squares` with a blue color.
            - Updates the display to reflect the changes.
        Note:
            This function assumes that `self.screen`, `self.CELL_WIDTH`, `self.CELL_HEIGHT`, and 
            `self.colors` are properly initialized attributes of the class.
        """

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
        
        if ants and hasattr(self, "ants"):
            for ant in self.ants:
                x, y = ant.location
                if ant.has_food:
                    pygame.draw.circle(self.screen, self.colors["busy_ants"], (x, y), 4)
                else:
                    pygame.draw.circle(self.screen, self.colors["empty_ants"], (x, y), 4)

        for square in squares:
            x, y = square
            rect = pygame.Rect(x * self.CELL_WIDTH, y * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
            pygame.draw.rect(self.screen, (0, 0, 255), rect)
        


        pygame.display.flip()

    def run(self):
        """
        Executes the main loop of the simulation.
        This method continuously processes events, updates the simulation state,
        and renders the simulation display. It listens for user input to quit
        the simulation and ensures proper cleanup of resources.
        Steps:
        1. Listens for Pygame events, such as quitting the application.
        2. Calls `self.iter_simulation()` to update the simulation state.
        3. Calls `self.display()` to render the simulation grids with their
           respective colors.
        Note:
            This method runs indefinitely until the user quits the application.
        Raises:
            SystemExit: Exits the program when the quit event is triggered.
        """
        logging.debug("Running simulation")

        # Main loop
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
                (self.food_grid, self.colors["food"]),
                (self.search_pheromone_grid, self.colors["search_pheromone"]),
                (self.food_pheromone_grid, self.colors["food_pheromone"])
            ])

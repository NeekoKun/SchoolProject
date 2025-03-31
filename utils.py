import numpy as np
import random

class Ant:
    def __init__(self, coordinates, rotation=None, speed=10, vision_radius=50, radius=5):
        self.steps = 0
        self.location = coordinates
        self.has_food = False
        self.vision_radius = vision_radius
        self.rotation = rotation if rotation else random.uniform(0, 2 * np.pi)
        self.speed = speed
        self.radius = radius
    
    def distance(self, other):
        return np.sqrt((self.location[0] - other[0]) ** 2 + (self.location[1] - other[1]) ** 2)

    def move(self, food, food_radius, colony, colony_radius, grid_size, pheromones: list[list[np.float64], float]):
        """
        "Move the ant towards food or colony based on its current state."
        """
        ## Rotate randomly
        if random.random() < 0.1:  # 10% chance to rotate randomly
            self.rotation += random.uniform(-np.pi / 4, np.pi / 4)
            self.rotation %= (2 * np.pi)
        
        self.steps += 1

        ## Check if on food or on colony
        if not self.has_food:
            for food_location in food:
                if self.distance(food_location) < food_radius + self.radius:
                    self.has_food = True
                    self.rotation += np.pi
                    self.steps = 0
                    return "Collected food"
        else:
            if self.distance(colony) < colony_radius + self.radius: 
                self.has_food = False
                self.rotation += np.pi
                self.steps = 0
                return "Deposited food"

        ## Check if food or colony in FOV
        targets = []
        targets_weight = []

        if not self.has_food: # Looking for food
            for location, strenght in pheromones:
                angle_to_food_pheromone = np.arctan2(location[1] - self.location[1], location[0] - self.location[0])
                angle_diff = (angle_to_food_pheromone - self.rotation + np.pi) % (2 * np.pi) - np.pi
                if abs(angle_diff) > np.pi / 4:
                    continue
                else:
                    targets.append(tuple(location))
                    targets_weight.append(strenght)

            for food_location in food:
                if self.distance(food_location) < self.vision_radius:
                    angle_to_food = np.arctan2(food_location[1] - self.location[1], food_location[0] - self.location[0])
                    angle_diff = (angle_to_food - self.rotation + np.pi) % (2 * np.pi) - np.pi
                    if abs(angle_diff) > np.pi / 4:  # Assuming a 90-degree cone of vision
                        continue
                    else:
                        targets.append(tuple(food_location))
                        targets_weight.append(100000)

        else: # Looking for home
            for location, strenght in pheromones:
                angle_to_home_pheromone = np.arctan2(location[1] - self.location[1], location[0] - self.location[0])
                angle_diff = (angle_to_home_pheromone - self.rotation + np.pi) % (2 * np.pi) - np.pi
                if abs(angle_diff) > np.pi / 4:
                    continue
                else:
                    targets.append(tuple(location))
                    targets_weight.append(strenght)
            
            if self.distance(colony) < self.vision_radius:
                angle_to_colony = np.arctan2(colony[1] - self.location[1], colony[0] - self.location[0])
                angle_diff = (angle_to_colony - self.rotation + np.pi) % (2 * np.pi) - np.pi
                if abs(angle_diff) > np.pi / 4:
                    return
                else:
                    targets = [tuple(colony)]
                    targets_weight = [1]

        if len(targets) > 0:
            total_weight = sum(targets_weight)
            targets_weight = [weight / total_weight for weight in targets_weight]
            target = targets[np.random.choice(len(targets), p=targets_weight)]

            self.rotation = np.arctan2(target[1] - self.location[1], target[0] - self.location[0])

        ## Move in the current direction
        new_x = self.location[0] + self.speed * np.cos(self.rotation)
        new_y = self.location[1] + self.speed * np.sin(self.rotation)
        
        # Check if out of bounds (assuming bounds are 0 <= x, y <= 100 for example)
        if not (0 <= new_x <= grid_size[0] and 0 <= new_y <= grid_size[1]):
            self.rotation = (self.rotation + np.pi) % (2 * np.pi)  # Turn 180 degrees
        else:
            self.location = (new_x, new_y)
        
        return "Moved"
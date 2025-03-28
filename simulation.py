from utils import LivingAnt
import random
import pygame

class ColonySim:
    def __init__(self):
        self.SIZE = self.WIDTH, self.HEIGHT = 1000, 1000

        pygame.display.init()
        self.screen = pygame.display.set_mode(self.SIZE)

    def iter_simulation(self):
        pass

    def display(self):
        pass

    def run(self):
        while True:
            self.iter_simulation()
            self.display()
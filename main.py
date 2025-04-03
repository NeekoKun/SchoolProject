import numpy as np
import pygame

class Body:
    def __init__(self, mass, radius, position, velocity):
        self.mass = mass
        self.radius = radius
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.location_track = []
        self.G = 100

    def distance(self, body):
        return np.linalg.norm(self.position - body.position)

    def stack_update(self, body_a, body_b, dt):
        self.location_track.append(self.position.copy())

        accelleration_a = self.G * (self.mass * body_a.mass / self.distance(body_a) ** 2) / self.mass
        accelleration_b = self.G * (self.mass * body_b.mass / self.distance(body_b) ** 2) / self.mass

        accelleration_a = (body_a.position - self.position) * accelleration_a
        accelleration_b = (body_b.position - self.position) * accelleration_b

        self.velocity += (accelleration_a + accelleration_b) * dt
        self.position += self.velocity * dt


SIZE = WIDTH, HEIGHT = 800, 600

pygame.init()
screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption("Simulation")
clock = pygame.time.Clock()
"""
r = 100
v = 10

body_a = Body(1.0, 10, (0, -r), (v, 0))
body_b = Body(1.0, 10, (r * np.cos(np.pi/6), r * np.sin(np.pi/6)), (-v * np.cos(np.pi/3), v * np.sin(np.pi/3)))
body_c = Body(1.0, 10, (-r * np.cos(np.pi/6), r * np.sin(np.pi/6)), (-v * np.cos(np.pi/3), -v * np.sin(np.pi/3)))
"""

body_a = Body(1.0, 10, (100, 0), (0, 200))
body_b = Body(1_000, 10, (0, 0), (0, 0))
body_c = Body(1.0, 10, (-100, 0), (0, -200))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    body_a.stack_update(body_b, body_c, 0.01)
    body_b.stack_update(body_a, body_c, 0.01)
    body_c.stack_update(body_a, body_b, 0.01)

    screen.fill((0, 0, 0))  # Clear the screen with black

    # Draw the bodies

    center = WIDTH // 2, HEIGHT // 2

    pygame.draw.circle(screen, (255, 0, 0), body_a.position.astype(int) + center, body_a.radius)
    pygame.draw.circle(screen, (0, 255, 0), body_b.position.astype(int) + center, body_b.radius)
    pygame.draw.circle(screen, (0, 0, 255), body_c.position.astype(int) + center, body_c.radius)

    for track_a, track_b, track_c in zip(body_a.location_track, body_b.location_track, body_c.location_track):
        pygame.draw.circle(screen, (200, 100, 100), track_a.astype(int) + center, 1)
        pygame.draw.circle(screen, (100, 200, 100), track_b.astype(int) + center, 1)
        pygame.draw.circle(screen, (100, 100, 200), track_c.astype(int) + center, 1)

    pygame.display.flip()

    clock.tick(60)  # Limit the frame rate to 10 FPS
import numpy as np
import pygame

def movement_equation(t, r):
    w = 0.05
    return np.array([r * np.cos(w * t), r * np.sin(w * t)])

pygame.init()

SIZE = WIDTH, HEIGHT = 1000, 1000
screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption("Movement Simulation")

running = True
clock = pygame.time.Clock()

center = np.array([WIDTH // 2, HEIGHT // 2])

t = 0

v_max = 10
follower = np.array([-100.0, 0.0])
target = movement_equation(t, 100)
target_distance = 20

while running:
    t += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    target = movement_equation(t, 100)

    if np.linalg.norm(target - follower) < target_distance:
        direction = (target - follower) / -np.linalg.norm(target - follower)
        follower += direction * v_max
    else:
        needed_velocity = np.linalg.norm(target - follower) - target_distance
        direction = (target - follower) / np.linalg.norm(target - follower)
        follower += direction * needed_velocity

    screen.fill((0, 0, 0))  # Clear screen with black
    
    pygame.draw.circle(screen, (200, 200, 200), follower + center, 10)  # Draw follower
    pygame.draw.circle(screen, (200, 200, 0), target + center, 5)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
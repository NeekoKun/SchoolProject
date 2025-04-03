import numpy as np
import pygame

SIZE = WIDTH, HEIGHT = 1000, 1000

RANGE = (-10, 10)

def equation(x, y, dt):
    vx = y+x
    vy = x/(y+1)**2
    return np.array([x + vx*dt, y + vy*dt])


pygame.display.init()
screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption("Differential Equation Renderer")
clock = pygame.time.Clock()
point_count = 1000

points = np.random.rand(point_count, 2) * (RANGE[1] - RANGE[0]) + RANGE[0]

def update(in_points):
    new_points = []
    for point in in_points:
        new_points.append(equation(point[0], point[1], 0.01))

    return new_points

def display(in_points, new_in_points):

    surface = pygame.Surface(SIZE, pygame.SRCALPHA)
    surface.fill((0, 0, 0, int(0.1 * 255)))
    screen.blit(surface, (0, 0))


    for point, new_point in zip(in_points, new_in_points):
        coors1 = point * (WIDTH / (RANGE[1] - RANGE[0])) + (WIDTH / 2, HEIGHT / 2)
        coors2 = new_point * (WIDTH / (RANGE[1] - RANGE[0])) + (WIDTH / 2, HEIGHT / 2)
        point = (int(coors1[0]), int(coors1[1]))
        new_point = (int(coors2[0]), int(coors2[1]))
        pygame.draw.line(screen, (255, 100, 100), point, new_point, 1)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    
    new_points = update(points)

    for i in range(point_count):
        if points[i][0] < RANGE[0] or points[i][0] > RANGE[1] or points[i][1] < RANGE[0] or points[i][1] > RANGE[1]:
            coors = np.random.rand(2) * (RANGE[1] - RANGE[0]) + RANGE[0]
            points[i] = coors
            new_points[i] = coors    

    display(points, new_points)

    points = np.array(new_points)

    clock.tick(30)
    pygame.display.flip()

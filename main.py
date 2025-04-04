import numpy as np
import pygame
import random


SIZE = WIDTH, HEIGHT = 1920, 1080

RANGE = (20, 20)
ORIGIN = (0, 0)

point_base_color = (255, 0, 0)
point_fast_color = (255, 255, 255)

point_r_diff = point_fast_color[0] - point_base_color[0]
point_g_diff = point_fast_color[1] - point_base_color[1]
point_b_diff = point_fast_color[2] - point_base_color[2]
point_color_diff = np.array([point_r_diff, point_g_diff, point_b_diff])

alpha = 1
beta = 1
delta = 1
gamma = 1

def equation(x, y, dt):
    global alpha, beta, delta, gamma
    vx = x * (alpha - beta * y)
    vy = y * (delta * x - gamma)
    return np.array([x + vx*dt, y + vy*dt])


pygame.display.init()
screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption("Differential Equation Renderer")
clock = pygame.time.Clock()
point_count = 1000

points = np.random.rand(point_count, 2) * (RANGE[1] - RANGE[0]) + RANGE[0]
tracking_points = []

def update(in_points):
    new_points = []
    for point in in_points:
        new_points.append(equation(point[0], point[1], 0.01))

    return new_points

def update_tracking_points(in_points):
    points = []
    for tracking_point in in_points:
        points.append(tracking_point)
        points[-1].append(equation(tracking_point[-1][0], tracking_point[-1][1], 0.001))
    return points

def display(in_points, new_in_points):

    surface = pygame.Surface(SIZE, pygame.SRCALPHA)
    surface.fill((0, 0, 0, int(0.1 * 255)))
    screen.blit(surface, (0, 0))


    for point, new_point in zip(in_points, new_in_points):
        coors1 = [point[0] * WIDTH / (RANGE[1] - RANGE[0]), HEIGHT - point[1] * HEIGHT / (RANGE[1] - RANGE[0])]
        coors2 = [point[0] * WIDTH / (RANGE[1] - RANGE[0]), HEIGHT - point[1] * HEIGHT / (RANGE[1] - RANGE[0])]
        speed = np.sqrt((coors1[0] - coors2[0])**2 + (coors1[1] - coors2[1])**2)
        point = (int(coors1[0]), int(coors1[1]))
        new_point = (int(coors2[0]), int(coors2[1]))
        color_coefficient = 1 - (1 / (np.sqrt(speed) + 1))
        color = point_base_color + color_coefficient * point_color_diff
        pygame.draw.line(screen, color, point, new_point, 1)

def display_tracking_points(in_tracking_points):
    for tracking_point in in_tracking_points:
        for i in range(len(tracking_point) - 1):
            coors1 = [tracking_point[i][0] * WIDTH / (RANGE[1] - RANGE[0]), HEIGHT - tracking_point[i][1] * HEIGHT / (RANGE[1] - RANGE[0])]
            coors2 = [tracking_point[i+1][0] * WIDTH / (RANGE[1] - RANGE[0]), HEIGHT - tracking_point[i+1][1] * HEIGHT / (RANGE[1] - RANGE[0])]
            speed = np.sqrt((coors1[0] - coors2[0])**2 + (coors1[1] - coors2[1])**2)
            point = (int(coors1[0]), int(coors1[1]))
            new_point = (int(coors2[0]), int(coors2[1]))
            pygame.draw.line(screen, (100, 200, 100), point, new_point, max(int(speed), 1))

def add_point(x, y):
    coors = [np.array([x, y])]

    tracking_points.append(coors)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            x = (x - WIDTH / 2) * (RANGE[1] - RANGE[0]) / WIDTH
            y = (y - HEIGHT / 2) * (RANGE[1] - RANGE[0]) / HEIGHT
            print(x, y)
            add_point(x, y)

    new_points = update(points)
    tracking_points = update_tracking_points(tracking_points)

    for i in range(point_count):
        if points[i][0] < RANGE[0] or points[i][0] > RANGE[1] or points[i][1] < RANGE[0] or points[i][1] > RANGE[1] or random.random() < 0.01:
            coors = np.random.rand(2) * (RANGE[1] - RANGE[0]) + RANGE[0]
            points[i] = coors
            new_points[i] = coors    

    display(points, new_points)
    display_tracking_points(tracking_points)

    points = np.array(new_points)

    clock.tick(30)
    pygame.display.flip()


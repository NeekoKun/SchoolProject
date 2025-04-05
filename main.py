import numpy as np
import pygame

SIZE = WIDTH, HEIGHT = 1000, 1000

RANGE = (10, 10)
ORIGIN = (0, 0)
PADDING = (1, 1)

for i in RANGE:
    if i < 0:
        raise ValueError("RANGE must be positive")

## Colors
axis_color = (255, 255, 255)

background_color = (10, 10, 10)
point_base_color = (255, 0, 0)
point_fast_color = (0, 0, 255)

point_r_diff = point_fast_color[0] - point_base_color[0]
point_g_diff = point_fast_color[1] - point_base_color[1]
point_b_diff = point_fast_color[2] - point_base_color[2]
point_color_diff = np.array([point_r_diff, point_g_diff, point_b_diff])

def equation(x, y, dt):
    dy = x-y
    dx = x+y
    return np.array([x + dx*dt, y + dy*dt])


pygame.display.init()
screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption("Differential Equation Renderer")
clock = pygame.time.Clock()
point_count = 10000
screen.fill(background_color)

background_layer = pygame.Surface(SIZE)
axis_layer       = pygame.Surface(SIZE, pygame.SRCALPHA)

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
        points[-1].append(equation(tracking_point[-1][0], tracking_point[-1][1], 0.01))
    return points

def display(in_points, new_in_points):
    global background_layer

    surface = pygame.Surface(SIZE, pygame.SRCALPHA)
    surface.fill((background_color[0], background_color[1], background_color[2], int(0.05 * 255)))
    background_layer.blit(surface, (0, 0))


    for point, new_point in zip(in_points, new_in_points):
        coors1 = [(point[0]     - ORIGIN[0]) * WIDTH / RANGE[0], WIDTH - (point[1]     - ORIGIN[1]) * HEIGHT / RANGE[1]]
        coors2 = [(new_point[0] - ORIGIN[0]) * WIDTH / RANGE[0], WIDTH - (new_point[1] - ORIGIN[1]) * HEIGHT / RANGE[1]]
        #coors1 = point * (WIDTH / (RANGE[1] - RANGE[0])) + (WIDTH / 2, HEIGHT / 2)
        #coors2 = new_point * (WIDTH / (RANGE[1] - RANGE[0])) + (WIDTH / 2, HEIGHT / 2)
        speed = np.sqrt((coors1[0] - coors2[0])**2 + (coors1[1] - coors2[1])**2)
        point = (int(coors1[0]), int(coors1[1]))
        new_point = (int(coors2[0]), int(coors2[1]))
        color_coefficient = 1 - (1 / (np.sqrt(speed) + 1))
        color = point_base_color + color_coefficient * point_color_diff
        pygame.draw.line(background_layer, color, point, new_point, 1)

def display_tracking_points(in_tracking_points):
    global background_layer

    for tracking_point in in_tracking_points:
        for i in range(len(tracking_point) - 1):
            coors1 = [(tracking_point[i][0]       - ORIGIN[0]) * WIDTH / RANGE[0], WIDTH - (tracking_point[i][1]       - ORIGIN[1]) * HEIGHT / RANGE[1]]
            coors2 = [(tracking_point[i+1][0]     - ORIGIN[0]) * WIDTH / RANGE[0], WIDTH - (tracking_point[i+1][1]     - ORIGIN[1]) * HEIGHT / RANGE[1]]
            speed = np.sqrt((coors1[0] - coors2[0])**2 + (coors1[1] - coors2[1])**2)
            point = (int(coors1[0]), int(coors1[1]))
            new_point = (int(coors2[0]), int(coors2[1]))
            pygame.draw.line(background_layer, (100, 200, 100), point, new_point, int(speed))

def add_point(x, y):
    coors = [np.array([x, y])]

    tracking_points.append(coors)

x0, y0 = 0, 0
x_shift, y_shift = 0, 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and pygame.key.get_pressed()[pygame.K_LSHIFT]: # LMB + Shift: Rudimentary panning
                x0, y0 = pygame.mouse.get_pos()

            elif event.button == 1:
                x, y = pygame.mouse.get_pos()
                x = (x / WIDTH) * RANGE[0] + ORIGIN[0]
                y = ((HEIGHT - y) / HEIGHT) * RANGE[1] + ORIGIN[1]
                print(x, y)
                add_point(x, y)
            if event.button == 4: # Scroll Up: Rudimentary zoom in
                RANGE = (RANGE[0] * 0.9, RANGE[1] * 0.9)
            if event.button == 5: # Scroll Down: Rudimentary zoom out
                RANGE = (RANGE[0] * 1.1, RANGE[1] * 1.1)
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                x0, y0 = 0, 0
        if pygame.mouse.get_pressed()[0] and pygame.key.get_pressed()[pygame.K_LSHIFT]: # LMB + Shift: Rudimentary panning
            x, y = pygame.mouse.get_pos()
            if x0 != 0 or y0 != 0:
                x_shift = (x - x0) / WIDTH * RANGE[0]
                y_shift = ((HEIGHT - y) - (HEIGHT - y0)) / HEIGHT * RANGE[1]
                ORIGIN = (ORIGIN[0] - x_shift, ORIGIN[1] - y_shift)
            x0, y0 = pygame.mouse.get_pos()

    new_points = update(points)
    tracking_points = update_tracking_points(tracking_points)

    # Check if points are out of bounds
    for i in range(point_count):
        if points[i][0] < ORIGIN[0] - PADDING[0] or points[i][0] > ORIGIN[0] + RANGE[1] + PADDING[0]  or points[i][1] < ORIGIN[1] - PADDING[1] or points[i][1] > ORIGIN[1] + RANGE[1] + PADDING[1]:
            coors = [np.random.rand() * (RANGE[0] + 2 * PADDING[0]) + (ORIGIN[0] - PADDING[0]), np.random.rand() * (RANGE[1] + 2 * PADDING[1]) + (ORIGIN[1] - PADDING[1])]
            points[i] = coors
            new_points[i] = coors

    # Display the points
    display(points, new_points)
    display_tracking_points(tracking_points)

    # Display the Axes
    axis_layer.fill((0, 0, 0, 0))
    if ORIGIN[0] < 0 < ORIGIN[0] + RANGE[0]:
        pygame.draw.line(axis_layer, axis_color, (-ORIGIN[0] * WIDTH / RANGE[0], 0), (-ORIGIN[0] * WIDTH / RANGE[0], HEIGHT), 1)
    elif ORIGIN[0] >= 0:
        pygame.draw.line(axis_layer, axis_color, (0, 0), (0, HEIGHT), 1)
    else:
        pygame.draw.line(axis_layer, axis_color, (WIDTH-1, 0), (WIDTH-1, HEIGHT), 1)
    if ORIGIN[1] < 0 < ORIGIN[1] + RANGE[1]:
        ax_height = HEIGHT - ORIGIN[1] * (-HEIGHT / RANGE[1]) - RANGE[1]
        pygame.draw.line(axis_layer, axis_color, (0, ax_height), (WIDTH, ax_height), 1)
    elif ORIGIN[1] >= 0:
        pygame.draw.line(axis_layer, axis_color, (0, HEIGHT-1), (WIDTH, HEIGHT-1), 1)
    else:
        pygame.draw.line(axis_layer, axis_color, (0, 0), (WIDTH, 0), 1)

    # Blit background and Axix
    screen.blit(background_layer, (0, 0))
    screen.blit(axis_layer, (0, 0))

    points = np.array(new_points)

    clock.tick(30)
    pygame.display.flip()

import numpy as np
import pygame

SIZE = WIDTH, HEIGHT = 1000, 1000

RANGE = [10, 10]
ORIGIN = [0, 0]
PADDING = (1, 1)
dt = 0.01
reset_probability = 0.1

for i in RANGE:
    if i < 0:
        raise ValueError("RANGE must be positive")

## Colors
axes_color = (200, 200, 200)

background_color = (10, 10, 10)
point_base_color = (255, 0, 0)
point_fast_color = (0, 0, 255)

point_r_diff = point_fast_color[0] - point_base_color[0]
point_g_diff = point_fast_color[1] - point_base_color[1]
point_b_diff = point_fast_color[2] - point_base_color[2]
point_color_diff = np.array([point_r_diff, point_g_diff, point_b_diff])

def equation(x, y, dx, dy):
    global dt
    d2y = 0
    d2x = 0
    dy = x
    dx = -y
    dy += d2y * dt
    dx += d2x * dt
    return np.array([x + dx*dt, y + dy*dt, dx, dy])

pygame.display.init()
screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption("Differential Equation Renderer")
clock = pygame.time.Clock()
pygame.font.init()
tick_font = pygame.font.SysFont('Arial', 15)
point_count = 10000
screen.fill(background_color)

background_layer = pygame.Surface(SIZE)
axes_layer       = pygame.Surface(SIZE, pygame.SRCALPHA)

dimming_surface = pygame.Surface(SIZE, pygame.SRCALPHA)
dimming_surface.fill((background_color[0], background_color[1], background_color[2], int(0.05 * 255)))

points = np.random.rand(point_count, 4)
points[:, 0] = points[:, 0] * RANGE[0] + ORIGIN[0]
points[:, 1] = points[:, 1] * RANGE[1] + ORIGIN[1]
tracking_points = []

def update(in_points):
    new_points = []
    for point in in_points:
        new_points.append(equation(point[0], point[1], point[2], point[3]))

    return new_points

def update_tracking_points(in_points):
    points = []
    for tracking_point in in_points:
        points.append(tracking_point)
        points[-1].append(equation(tracking_point[-1][0], tracking_point[-1][1], tracking_point[-1][2], tracking_point[-1][3]))
    return points

def get_draw_data(point, new_point):
    coors1 = [(point[0]     - ORIGIN[0]) * WIDTH / RANGE[0], WIDTH - (point[1]     - ORIGIN[1]) * HEIGHT / RANGE[1]]
    coors2 = [(new_point[0] - ORIGIN[0]) * WIDTH / RANGE[0], WIDTH - (new_point[1] - ORIGIN[1]) * HEIGHT / RANGE[1]]
    speed = np.sqrt((coors1[0] - coors2[0])**2 + (coors1[1] - coors2[1])**2)
    point = (int(coors1[0]), int(coors1[1]))
    new_point = (int(coors2[0]), int(coors2[1]))
    color_coefficient = 1 - (1 / (np.sqrt(speed) + 1))
    color = point_base_color + color_coefficient * point_color_diff
    return coors1, coors2, color

def display(in_points, new_in_points):
    global background_layer, dimming_surface

    background_layer.blit(dimming_surface, (0, 0))

    for point, new_point in zip(in_points, new_in_points):
        coors1, coors2, color = get_draw_data(point, new_point) # TODO: Parallelize this
        if (color != point_base_color).any():
            pygame.draw.line(background_layer, color, coors1, coors2, 1)

def display_tracking_points(in_tracking_points):
    global background_layer

    for tracking_point in in_tracking_points:
        for i in range(len(tracking_point) - 1):
            coors1 = [(tracking_point[i][0]       - ORIGIN[0]) * WIDTH / RANGE[0], WIDTH - (tracking_point[i][1]       - ORIGIN[1]) * HEIGHT / RANGE[1]]
            coors2 = [(tracking_point[i+1][0]     - ORIGIN[0]) * WIDTH / RANGE[0], WIDTH - (tracking_point[i+1][1]     - ORIGIN[1]) * HEIGHT / RANGE[1]]
            speed = np.sqrt((coors1[0] - coors2[0])**2 + (coors1[1] - coors2[1])**2)
            point = (int(coors1[0]), int(coors1[1]))
            new_point = (int(coors2[0]), int(coors2[1]))
            pygame.draw.line(background_layer, (100, 200, 100), point, new_point, max(int(speed), 1))

def coors_to_screen(coors):
    coors = np.array(coors)
    coors[0] = (coors[0] - ORIGIN[0]) * WIDTH / RANGE[0]
    coors[1] = HEIGHT - (coors[1] - ORIGIN[1]) * HEIGHT / RANGE[1]
    return coors

def round_to_nearest(value, step):
    return int(round(value + step - 1) // step * step)

def display_axes():
    # Display the Axes
    axes_layer.fill((0, 0, 0, 0))
    x_ax_tick_thickness = 6
    y_ax_tick_thickness = 6

    if ORIGIN[0] < 0 < ORIGIN[0] + RANGE[0]:
        ax_width = - ORIGIN[0] * WIDTH / RANGE[0]
    elif ORIGIN[0] >= 0:
        ax_width = 0
    else:
        ax_width = WIDTH-1

    if ORIGIN[1] < 0 < ORIGIN[1] + RANGE[1]:
        ax_height = ((ORIGIN[1] + RANGE[1]) * HEIGHT / RANGE[1])
    elif ORIGIN[1] >= 0:
        ax_height = HEIGHT-1
    else:
        ax_height = 0
    
    pygame.draw.line(axes_layer, axes_color, (ax_width, 0), (ax_width, HEIGHT), 3)
    pygame.draw.line(axes_layer, axes_color, (0, ax_height), (WIDTH, ax_height), 3)

    # Display Axes scale
    step = max(RANGE[0] // 10, 1)

    x_tick_coordinates = [ i * step + round_to_nearest(ORIGIN[0], step)  for i in range(int(np.ceil(RANGE[0] / step)))]
    y_tick_coordinates = [ i * step + round_to_nearest(ORIGIN[1], step) for i in range(int(np.ceil(RANGE[1] / step)))]

    for i in x_tick_coordinates: # Calculate x axis ticks
        coors = coors_to_screen([i, 0])
        pygame.draw.line(axes_layer, axes_color, (coors[0], min(max(coors[1], 0), HEIGHT)  - x_ax_tick_thickness), (coors[0], min(max(coors[1], 0), HEIGHT) + x_ax_tick_thickness), 2)
        if i != 0:
            text = tick_font.render(str(i), True, axes_color)
            text_rect = text.get_rect(center=(coors[0], coors[1] + 10))
            if ax_height < HEIGHT / 2:
                axes_layer.blit(text, (coors[0] - text_rect.width / 2, ax_height + 30))
            else:
                axes_layer.blit(text, (coors[0] - text_rect.width / 2, ax_height - 30))

    for i in y_tick_coordinates: # Calculate y axis ticks
        coors = coors_to_screen([0, i])
        pygame.draw.line(axes_layer, axes_color, (min(max(coors[0], 0), WIDTH) - y_ax_tick_thickness, coors[1]), (min(max(coors[0], 0), WIDTH)  + y_ax_tick_thickness, coors[1]), 2)
        if i != 0:
            text = tick_font.render(str(i), True, axes_color)
            text_rect = text.get_rect(center=(coors[0] + 10, coors[1]))
            if ax_width > WIDTH / 2:
                axes_layer.blit(text, (ax_width - text_rect.width - 20, coors[1] - text_rect.height / 2))
            else:
                axes_layer.blit(text, (ax_width + 20, coors[1] - text_rect.height / 2))

    zero_text = tick_font.render("0", True, axes_color)
    if ax_width > WIDTH / 2 and ax_height < HEIGHT / 2:
        axes_layer.blit(zero_text, (ax_width - zero_text.get_width() - 20, ax_height + 30))
    elif ax_width > WIDTH / 2 and ax_height > HEIGHT / 2:
        axes_layer.blit(zero_text, (ax_width - zero_text.get_width() - 20, ax_height - 30))
    elif ax_width < WIDTH / 2 and ax_height < HEIGHT / 2:
        axes_layer.blit(zero_text, (ax_width + 20, ax_height + 30))
    elif ax_width < WIDTH / 2 and ax_height > HEIGHT / 2:   
        axes_layer.blit(zero_text, (ax_width + 20, ax_height - 30))

def add_point(x, y):
    coors = [np.array([x, y, 0, 0])]

    tracking_points.append(coors)

x0, y0 = 0, 0
x_shift, y_shift = 0, 0
moved = True

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                x, y = pygame.mouse.get_pos()
                x = (x / WIDTH) * RANGE[0] + ORIGIN[0]
                y = ((HEIGHT - y) / HEIGHT) * RANGE[1] + ORIGIN[1]
                add_point(x, y)
            if event.button == 2: # Middle Mouse Button: Panning
                x0, y0 = pygame.mouse.get_pos()
            if event.button == 4: # Scroll Up: Zoom in
                """
                Zoom In:
                The Range beecomes 90% of the original range
                """
                zoom_x, zoom_y = pygame.mouse.get_pos()
                ORIGIN[0] += (RANGE[0] / 10) * (zoom_x / WIDTH)
                ORIGIN[1] += (RANGE[1] / 10) * ((HEIGHT - zoom_y) / HEIGHT)
                RANGE = [RANGE[0] * 0.9, RANGE[1] * 0.9]
                moved = True
            if event.button == 5: # Scroll Down: Zoom out
                """
                Zoom In:
                The Range beecomes 110% of the original range
                """
                zoom_x, zoom_y = pygame.mouse.get_pos()
                ORIGIN[0] -= (RANGE[0] / 10) * (zoom_x / WIDTH)
                ORIGIN[1] -= (RANGE[1] / 10) * ((HEIGHT - zoom_y) / HEIGHT)
                RANGE = [RANGE[0] * 1.1, RANGE[1] * 1.1]
                moved = True
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 2:
                x0, y0 = 0, 0
        if pygame.mouse.get_pressed()[1]: # LMB + Shift: Rudimentary panning
            x, y = pygame.mouse.get_pos()
            if x0 != 0 or y0 != 0:
                x_shift = (x - x0) / WIDTH * RANGE[0]
                y_shift = ((HEIGHT - y) - (HEIGHT - y0)) / HEIGHT * RANGE[1]
                ORIGIN = [ORIGIN[0] - x_shift, ORIGIN[1] - y_shift]
            x0, y0 = pygame.mouse.get_pos()
            moved = True

    new_points = update(points)
    tracking_points = update_tracking_points(tracking_points)

    # Check if points are out of bounds
    for i in range(point_count):
        if points[i][0] < ORIGIN[0] - PADDING[0] or points[i][0] > ORIGIN[0] + RANGE[1] + PADDING[0]  or points[i][1] < ORIGIN[1] - PADDING[1] or points[i][1] > ORIGIN[1] + RANGE[1] + PADDING[1] or np.random.rand() < reset_probability:
            coors = [np.random.rand() * (RANGE[0] + 2 * PADDING[0]) + (ORIGIN[0] - PADDING[0]), np.random.rand() * (RANGE[1] + 2 * PADDING[1]) + (ORIGIN[1] - PADDING[1]), 0, 0]
            points[i] = coors
            new_points[i] = coors

    # Display the points
    display(points, new_points)
    display_tracking_points(tracking_points)
    if moved:
        display_axes()
        moved = False

    # Blit background and Axix
    screen.blit(background_layer, (0, 0))
    screen.blit(axes_layer, (0, 0))

    points = np.array(new_points)


    clock.tick(60)
    pygame.display.flip()
    print("FPS: ", round(clock.get_fps(), 2), end="\r")

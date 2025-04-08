from sympy import symbols, diff, lambdify, sin, cos
import multiprocessing as mp
import numpy as np
import pygame
import json

class DifferentialEquationRenderer:
    def __init__(self, settings=None):
        self.initialize_settings(settings)

        self.t = 0
        self.points = np.random.rand(self.point_count, 4)
        self.points[:, 0] = self.points[:, 0] * self.RANGE[0] + self.ORIGIN[0]
        self.points[:, 1] = self.points[:, 1] * self.RANGE[1] + self.ORIGIN[1]
        self.tracking_points = []

        # Initialize di Vector Field equations
        self.x, self.y, self.t = symbols('x y t')

        self.P = self.x * sin(self.t / 10)
        self.Q = self.y * sin(self.t / 10)

        self.dPdx = lambdify((self.x, self.y, self.t), self.P)
        self.dQdy = lambdify((self.x, self.y, self.t), self.Q)

        self.curl = lambdify((self.x, self.y, self.t), diff(self.Q, self.x) - diff(self.P, self.y))
        self.divergence = lambdify((self.x, self.y, self.t), diff(self.P, self.x) + diff(self.Q, self.y))

    def initialize_settings(self, settings):
        if settings is None:
            settings = "settings.json"
        with open(settings, 'r') as f:
            data = json.load(f)
            self.title = data["title"]

            self.SIZE = self.WIDTH, self.HEIGHT = data["screen"]["width"], data["screen"]["height"]
            self.RANGE = [data["screen"]["width"] / data["screen"]["scale"][0], data["screen"]["height"] / data["screen"]["scale"][1]]
            self.ORIGIN = [data["screen"]["origin"][0] - self.RANGE[0] / 2, data["screen"]["origin"][1] - self.RANGE[1] / 2]
            self.PADDING = data["screen"]["padding"]

            self.dt = data["simulation"]["time_step"]
            self.reset_probability = data["simulation"]["point_reset_probability"]
            self.point_count = data["simulation"]["point_count"]
            self.curl_resolution = data["simulation"]["curl_resolution"]

            self.background_color = data["colors"]["background"]
            self.dimming_factor = data["colors"]["dimming_factor"]
            self.point_base_color = data["colors"]["base_point"]
            self.point_fast_color = data["colors"]["fast_point"]
            self.axes_color = data["colors"]["axes"]
            self.tracking_point_color = data["colors"]["tracking_point"]

        # Pygame Settings
        pygame.display.init()
        pygame.display.set_caption("Differential Equation Renderer")
        pygame.font.init()

        self.screen = pygame.display.set_mode(self.SIZE)
        self.clock = pygame.time.Clock()
        
        self.background_layer = pygame.Surface(self.SIZE, pygame.SRCALPHA)
        self.curl_layer       = pygame.Surface(self.SIZE, pygame.SRCALPHA)
        self.divergence_layer = pygame.Surface(self.SIZE, pygame.SRCALPHA)
        self.axes_layer       = pygame.Surface(self.SIZE, pygame.SRCALPHA)
        self.tracking_layer   = pygame.Surface(self.SIZE, pygame.SRCALPHA)
        self.UI_layer         = pygame.Surface(self.SIZE, pygame.SRCALPHA)
        self.dimming_surface  = pygame.Surface(self.SIZE, pygame.SRCALPHA)

        # UI stuff
        self.tick_font = pygame.font.SysFont('Arial', 15)
        self.UI_font   = pygame.font.SysFont('Arial', 20)
        self.pause_button = pygame.Rect(self.WIDTH - 50, 10, 40, 40)

        # Colors
        self.point_r_diff = self.point_fast_color[0] - self.point_base_color[0]
        self.point_g_diff = self.point_fast_color[1] - self.point_base_color[1]
        self.point_b_diff = self.point_fast_color[2] - self.point_base_color[2]
        self.point_color_diff = np.array([self.point_r_diff, self.point_g_diff, self.point_b_diff])
        
        self.screen.fill(self.background_color)
        self.dimming_surface.fill((self.background_color[0], self.background_color[1], self.background_color[2], int(self.dimming_factor * 255)))
    
        self.frame = 0


    def equation(self, x, y, dx, dy):
        dx = self.dPdx(x, y, self.frame)
        dy = self.dQdy(x, y, self.frame)
        return np.array([x + dx*self.dt, y + dy*self.dt, dx, dy])

    def update_curl(self):
        self.curl_layer.fill((0, 0, 0, 0))

        for i in range(self.curl_resolution[0]):
            for j in range(self.curl_resolution[1]):
                x = i * (self.RANGE[0] / self.curl_resolution[0]) + self.ORIGIN[0]
                y = j * (self.RANGE[1] / self.curl_resolution[1]) + self.ORIGIN[1]
                curl = int(255 * np.arctan(self.curl(x, y, self.frame)) / np.pi)
                coors = [x, y, 0, 0]
                coors = self.coors_to_screen(coors)
                color = (curl, 0, 0) if curl > 0 else (0, 0, -curl)
                pygame.draw.circle(self.curl_layer, color, (int(coors[0]), int(coors[1])), 2)

    def update_divergence(self):
        self.divergence_layer.fill((0, 0, 0, 0))

        for i in range(self.curl_resolution[0]):
            for j in range(self.curl_resolution[1]):
                x = i * (self.RANGE[0] / self.curl_resolution[0]) + self.ORIGIN[0]
                y = j * (self.RANGE[1] / self.curl_resolution[1]) + self.ORIGIN[1]
                divergence = int(255 * np.arctan(self.divergence(x, y, self.frame)) / np.pi)
                coors = [x, y, 0, 0]
                coors = self.coors_to_screen(coors)
                color = (divergence, 0, 0) if divergence > 0 else (0, 0, -divergence)
                pygame.draw.circle(self.divergence_layer, color, (int(coors[0] + self.WIDTH / (2 * self.curl_resolution[0])), int(coors[1] + self.WIDTH / (2 *self.curl_resolution[1]))), 2)

    def update(self, in_points):
        new_points = []
        for point in in_points:
            new_points.append(self.equation(point[0], point[1], point[2], point[3]))

        return new_points

    def update_tracking_points(self, in_points):
        points = []
        for tracking_point in in_points:
            points.append(tracking_point)
            points[-1].append(self.equation(tracking_point[-1][0], tracking_point[-1][1], tracking_point[-1][2], tracking_point[-1][3]))
        return points

    def get_draw_data(self, point, new_point):
        coors1 = [
            (point[0]     - self.ORIGIN[0]) * self.WIDTH / self.RANGE[0],
            self.HEIGHT - (point[1]     - self.ORIGIN[1]) * self.HEIGHT / self.RANGE[1]
        ]
        coors2 = [
            (new_point[0] - self.ORIGIN[0]) * self.WIDTH / self.RANGE[0],
            self.HEIGHT - (new_point[1] - self.ORIGIN[1]) * self.HEIGHT / self.RANGE[1]
        ]
        speed = np.sqrt((coors1[0] - coors2[0])**2 + (coors1[1] - coors2[1])**2)
        point = (int(coors1[0]), int(coors1[1]))
        new_point = (int(coors2[0]), int(coors2[1]))
        color_coefficient = 1 - (1 / (np.sqrt(speed) + 1))
        color = self.point_base_color + color_coefficient * self.point_color_diff
        return coors1, coors2, color, speed

    def coors_to_screen(self, coors):
        coors = np.array(coors)
        coors[0] = (coors[0] - self.ORIGIN[0]) * self.WIDTH / self.RANGE[0]
        coors[1] = self.HEIGHT - (coors[1] - self.ORIGIN[1]) * self.HEIGHT / self.RANGE[1]
        return coors

    def round_to_nearest(self, value, step):
        return int(round(value + step - 1) // step * step)

    def display(self, in_points, new_in_points):
        self.background_layer.blit(self.dimming_surface, (0, 0))

        for point, new_point in zip(in_points, new_in_points):
            coors1, coors2, color, speed = self.get_draw_data(point, new_point) # TODO: Parallelize this
            if speed != 0:
                pygame.draw.line(self.background_layer, color, coors1, coors2, 1)

    def display_tracking_points(self, in_tracking_points):
        self.tracking_layer.fill((0, 0, 0, 0))

        for tracking_point in in_tracking_points:
            for i in range(len(tracking_point) - 1):
                coors1, coors2, color, speed = self.get_draw_data(tracking_point[i], tracking_point[i+1])
                pygame.draw.line(self.tracking_layer, self.tracking_point_color, coors1, coors2, max(int(speed), 1))

    def display_axes(self):
        # Display the Axes
        self.axes_layer.fill((0, 0, 0, 0))
        x_ax_tick_thickness = 6
        y_ax_tick_thickness = 6

        if self.ORIGIN[0] < 0 < self.ORIGIN[0] + self.RANGE[0]:
            ax_width = - self.ORIGIN[0] * self.WIDTH / self.RANGE[0]
        elif self.ORIGIN[0] >= 0:
            ax_width = 0
        else:
            ax_width = self.WIDTH-1

        if self.ORIGIN[1] < 0 < self.ORIGIN[1] + self.RANGE[1]:
            ax_height = ((self.ORIGIN[1] + self.RANGE[1]) * self.HEIGHT / self.RANGE[1])
        elif self.ORIGIN[1] >= 0:
            ax_height = self.HEIGHT-1
        else:
            ax_height = 0
        
        pygame.draw.line(self.axes_layer, self.axes_color, (ax_width, 0), (ax_width, self.HEIGHT), 3)
        pygame.draw.line(self.axes_layer, self.axes_color, (0, ax_height), (self.WIDTH, ax_height), 3)

        # Display Axes scale
        step = max(self.RANGE[0] // 10, 1)

        x_tick_coordinates = [ i * step + self.round_to_nearest(self.ORIGIN[0], step)  for i in range(int(np.ceil(self.RANGE[0] / step)))]
        y_tick_coordinates = [ i * step + self.round_to_nearest(self.ORIGIN[1], step) for i in range(int(np.ceil(self.RANGE[1] / step)))]

        for i in x_tick_coordinates: # Calculate x axis ticks
            coors = self.coors_to_screen([i, 0])
            pygame.draw.line(self.axes_layer, self.axes_color, (coors[0], min(max(coors[1], 0), self.HEIGHT)  - x_ax_tick_thickness), (coors[0], min(max(coors[1], 0), self.HEIGHT) + x_ax_tick_thickness), 2)
            if i != 0:
                text = self.tick_font.render(str(int(i)), True, self.axes_color)
                text_rect = text.get_rect(center=(coors[0], coors[1] + 10))
                if ax_height < self.HEIGHT / 2:
                    self.axes_layer.blit(text, (coors[0] - text_rect.width / 2, ax_height + 30))
                else:
                    self.axes_layer.blit(text, (coors[0] - text_rect.width / 2, ax_height - 30))

        for i in y_tick_coordinates: # Calculate y axis ticks
            coors = self.coors_to_screen([0, i])
            pygame.draw.line(self.axes_layer, self.axes_color, (min(max(coors[0], 0), self.WIDTH) - y_ax_tick_thickness, coors[1]), (min(max(coors[0], 0), self.WIDTH)  + y_ax_tick_thickness, coors[1]), 2)
            if i != 0:
                text = self.tick_font.render(str(int(i)), True, self.axes_color)
                text_rect = text.get_rect(center=(coors[0] + 10, coors[1]))
                if ax_width > self.WIDTH / 2:
                    self.axes_layer.blit(text, (ax_width - text_rect.width - 20, coors[1] - text_rect.height / 2))
                else:
                    self.axes_layer.blit(text, (ax_width + 20, coors[1] - text_rect.height / 2))

        zero_text = self.tick_font.render("0", True, self.axes_color)
        if ax_width > self.WIDTH / 2 and ax_height < self.HEIGHT / 2:
            self.axes_layer.blit(zero_text, (ax_width - zero_text.get_width() - 20, ax_height + 30))
        elif ax_width > self.WIDTH / 2 and ax_height > self.HEIGHT / 2:
            self.axes_layer.blit(zero_text, (ax_width - zero_text.get_width() - 20, ax_height - 30))
        elif ax_width < self.WIDTH / 2 and ax_height < self.HEIGHT / 2:
            self.axes_layer.blit(zero_text, (ax_width + 20, ax_height + 30))
        elif ax_width < self.WIDTH / 2 and ax_height > self.HEIGHT / 2:   
            self.axes_layer.blit(zero_text, (ax_width + 20, ax_height - 30))

    def display_display_status(self):
        self.UI_layer.fill((0, 0, 0, 0))
        fps_text = self.tick_font.render(f"FPS: {round(self.clock.get_fps(), 2)}", True, (255, 255, 255))

        if self.show_divergence:
            divergence_text = self.tick_font.render("Divergence: ON", True, (0, 255, 0))
        else:
            divergence_text = self.tick_font.render("Divergence: OFF", True, (255, 0, 0))

        if self.show_curl:
            curl_text = self.tick_font.render("Curl: ON", True, (0, 255, 0))
        else:
            curl_text = self.tick_font.render("Curl: OFF", True, (255, 0, 0))

        curl_text_rect = curl_text.get_rect(topleft=(10, 10))
        divergence_text_rect = divergence_text.get_rect(topleft=(curl_text_rect.width + 20, 10))
        fps_text_rect = fps_text.get_rect(center=(self.WIDTH - 100, 20))

        self.UI_layer.blit(curl_text, curl_text_rect)
        self.UI_layer.blit(divergence_text, divergence_text_rect)
        self.UI_layer.blit(fps_text, fps_text_rect)

        if self.frame % 60 == 0:
            print("FPS: ", round(self.clock.get_fps(), 2), end="\r")



    def add_point(self, x, y):
        coors = [np.array([x, y, 0, 0])]

        self.tracking_points.append(coors)

    def run(self):
        x0, y0 = 0, 0
        x_shift, y_shift = 0, 0
        moved = True
        paused = False
        show_ui = True
        self.show_curl = True
        self.show_divergence = True

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h:
                        show_ui = not show_ui
                    elif event.key == pygame.K_o: ## Center on Origin
                        self.ORIGIN = [-self.RANGE[0] / 2, -self.RANGE[1] / 2]
                        moved = True
                    elif event.key == pygame.K_c: ## Show Curl
                        self.show_curl = not self.show_curl
                    elif event.key == pygame.K_d: ## Show Divergence
                        self.show_divergence = not self.show_divergence
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and self.pause_button.collidepoint(event.pos):
                        paused = not paused
                    elif event.button == 1:
                        x, y = pygame.mouse.get_pos()
                        x = (x / self.WIDTH) * self.RANGE[0] + self.ORIGIN[0]
                        y = ((self.HEIGHT - y) / self.HEIGHT) * self.RANGE[1] + self.ORIGIN[1]
                        self.add_point(x, y)
                    if event.button == 2 and not paused: # Middle Mouse Button: Panning
                        x0, y0 = pygame.mouse.get_pos()
                    if event.button == 4 and not paused: # Scroll Up: Zoom in
                        """
                        Zoom In:
                        The Range beecomes 90% of the original range
                        """
                        zoom_x, zoom_y = pygame.mouse.get_pos()
                        self.ORIGIN[0] += (self.RANGE[0] / 10) * (zoom_x / self.WIDTH)
                        self.ORIGIN[1] += (self.RANGE[1] / 10) * ((self.HEIGHT - zoom_y) / self.HEIGHT)
                        self.RANGE = [self.RANGE[0] * 0.9, self.RANGE[1] * 0.9]
                        self.PADDING = [self.PADDING[0] * 0.9, self.PADDING[1] * 0.9]
                        moved = True
                    if event.button == 5 and not paused: # Scroll Down: Zoom out
                        """
                        Zoom Out:
                        The Range beecomes 110% of the original range
                        """
                        zoom_x, zoom_y = pygame.mouse.get_pos()
                        self.ORIGIN[0] -= (self.RANGE[0] / 10) * (zoom_x / self.WIDTH)
                        self.ORIGIN[1] -= (self.RANGE[1] / 10) * ((self.HEIGHT - zoom_y) / self.HEIGHT)
                        self.RANGE = [self.RANGE[0] * 1.1, self.RANGE[1] * 1.1]
                        self.PADDING = [self.PADDING[0] * 1.1, self.PADDING[1] * 1.1]
                        moved = True
                if event.type == pygame.MOUSEBUTTONUP and not paused:
                    if event.button == 2:
                        x0, y0 = 0, 0
                if pygame.mouse.get_pressed()[1] and not paused: # LMB + Shift: Rudimentary panning
                    x, y = pygame.mouse.get_pos()
                    if x0 != 0 or y0 != 0:
                        x_shift = (x - x0) / self.WIDTH * self.RANGE[0]
                        y_shift = ((self.HEIGHT - y) - (self.HEIGHT - y0)) / self.HEIGHT * self.RANGE[1]
                        self.ORIGIN = [self.ORIGIN[0] - x_shift, self.ORIGIN[1] - y_shift]
                    x0, y0 = pygame.mouse.get_pos()
                    moved = True

            if not paused:
                new_points = self.update(self.points)
                self.tracking_points = self.update_tracking_points(self.tracking_points)

                # Check if points are out of bounds
                for i in range(self.point_count):
                    if self.points[i][0] < self.ORIGIN[0] - self.PADDING[0] or self.points[i][0] > self.ORIGIN[0] + self.RANGE[0] + self.PADDING[0]  or self.points[i][1] < self.ORIGIN[1] - self.PADDING[1] or self.points[i][1] > self.ORIGIN[1] + self.RANGE[1] + self.PADDING[1] or np.random.rand() < self.reset_probability:
                        coors = [np.random.rand() * (self.RANGE[0] + 2 * self.PADDING[0]) + (self.ORIGIN[0] - self.PADDING[0]), np.random.rand() * (self.RANGE[1] + 2 * self.PADDING[1]) + (self.ORIGIN[1] - self.PADDING[1]), 0, 0]
                        self.points[i] = coors
                        new_points[i] = coors

                # Display the points
                self.display(self.points, new_points)
                self.display_tracking_points(self.tracking_points)
                if moved:
                    if self.show_curl:
                        self.update_curl()
                    if self.show_divergence:
                        self.update_divergence()
                    self.display_axes()
                    #moved = False
                self.display_display_status()

            # Blit background and Axix
            self.screen.blit(self.background_layer, (0, 0))
            if self.show_curl:
                self.screen.blit(self.curl_layer, (0, 0))
            if self.show_divergence:
                self.screen.blit(self.divergence_layer, (0, 0))
            if show_ui:
                self.screen.blit(self.axes_layer, (0, 0))
                self.screen.blit(self.tracking_layer, (0, 0))
                self.screen.blit(self.UI_layer, (0, 0))
                pygame.draw.rect(self.screen, (200, 200, 200), self.pause_button) if paused else pygame.draw.rect(self.screen, (100, 100, 100), self.pause_button)

            self.points = np.array(new_points)

            self.clock.tick(60)
            self.frame += 1
            pygame.display.flip()

if __name__ == "__main__":
    solver = DifferentialEquationRenderer()
    solver.run()
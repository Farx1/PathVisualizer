import pygame
import sys
import math
from queue import PriorityQueue, Queue, LifoQueue
from tkinter import *
from tkinter import ttk
from tkinter import messagebox

# Constants
SCREEN_SIZE = 800
SIDE_PANEL_WIDTH = 350
BACKGROUND_COLOR = (240, 240, 240)
GRID_COLOR = (200, 200, 200)
START_COLOR = (0, 255, 0)
END_COLOR = (255, 0, 0)
PATH_COLOR = (0, 0, 255)
OPEN_COLOR = (0, 255, 255)
CLOSED_COLOR = (255, 165, 0)
OBSTACLE_COLOR = (0, 0, 0)
TEXT_COLOR = (50, 50, 50)
BUTTON_COLOR = (100, 100, 200)
BUTTON_HOVER_COLOR = (150, 150, 255)
BUTTON_TEXT_COLOR = (255, 255, 255)
PANEL_TITLE_COLOR = (0, 100, 200)
PANEL_SUBTITLE_COLOR = (0, 150, 200)

# Initialize Pygame
pygame.init()
pygame.font.init()

# Set the app icon
#icon = pygame.image.load('C:/Users/Sayad-Barth Jules/Desktop/Loupe.png')  # or .ico file
#pygame.display.set_icon(icon)

# Then create your window
#WIN = pygame.display.set_mode((SCREEN_SIZE + SIDE_PANEL_WIDTH, SCREEN_SIZE))
#pygame.display.set_caption("Pathfinding Algorithm Visualizer")

class Node:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = col * width
        self.y = row * width
        self.color = BACKGROUND_COLOR
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == CLOSED_COLOR

    def is_open(self):
        return self.color == OPEN_COLOR

    def is_obstacle(self):
        return self.color == OBSTACLE_COLOR

    def is_start(self):
        return self.color == START_COLOR

    def is_end(self):
        return self.color == END_COLOR

    def reset(self):
        self.color = BACKGROUND_COLOR

    def make_start(self):
        self.color = START_COLOR

    def make_closed(self):
        self.color = CLOSED_COLOR

    def make_open(self):
        self.color = OPEN_COLOR

    def make_obstacle(self):
        self.color = OBSTACLE_COLOR

    def make_end(self):
        self.color = END_COLOR

    def make_path(self):
        self.color = PATH_COLOR

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_obstacle():  # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])
        if self.row > 0 and not grid[self.row - 1][self.col].is_obstacle():  # UP
            self.neighbors.append(grid[self.row - 1][self.col])
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_obstacle():  # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])
        if self.col > 0 and not grid[self.row][self.col - 1].is_obstacle():  # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from, current, draw, start, end):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]

    path_length = len(path)  # Calculate the path length

    for node in reversed(path):
        if node != start and node != end:
            node.make_path()
            draw()

    start.make_start()
    end.make_end()
    draw()

    return path_length  # Return the path length

def a_star(draw, grid, start, end, win, instruction_surface, brush_size):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    start_time = pygame.time.get_ticks()

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            path_length = reconstruct_path(came_from, end, draw,start,end)
            end_time = pygame.time.get_ticks()
            algorithm_time = (end_time - start_time) / 1000
            update_instruction_surface(instruction_surface, brush_size, algorithm_time, path_length)
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()
        win.blit(instruction_surface, (SCREEN_SIZE, 0))
        pygame.display.update()

        if current != start and current != end:
            current.make_closed()
    return False

def dijkstra(draw, grid, start, end, win, instruction_surface, brush_size):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0

    open_set_hash = {start}

    start_time = pygame.time.get_ticks()

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            path_length = reconstruct_path(came_from, end, draw,start,end)
            end_time = pygame.time.get_ticks()
            algorithm_time = (end_time - start_time) / 1000
            update_instruction_surface(instruction_surface, brush_size, algorithm_time, path_length)
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((g_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()
        win.blit(instruction_surface, (SCREEN_SIZE, 0))
        pygame.display.update()

        if current != start and current != end:
            current.make_closed()

    return False

def bfs(draw, grid, start, end, win, instruction_surface, brush_size):
    queue = Queue()
    queue.put(start)
    came_from = {}
    visited = {start}

    start_time = pygame.time.get_ticks()

    while not queue.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        current = queue.get()

        if current == end:
            path_length = reconstruct_path(came_from, end, draw,start,end)
            end_time = pygame.time.get_ticks()
            algorithm_time = (end_time - start_time) / 1000
            update_instruction_surface(instruction_surface, brush_size, algorithm_time, path_length)
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.put(neighbor)
                neighbor.make_open()

        draw()
        win.blit(instruction_surface, (SCREEN_SIZE, 0))
        pygame.display.update()

        if current != start and current != end:
            current.make_closed()

    return False

def dfs(draw, grid, start, end, win, instruction_surface, brush_size):
    stack = LifoQueue()
    stack.put(start)
    came_from = {}
    visited = {start}

    start_time = pygame.time.get_ticks()

    while not stack.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        current = stack.get()

        if current == end:
            path_length = reconstruct_path(came_from, end, draw, start, end)
            end_time = pygame.time.get_ticks()
            algorithm_time = (end_time - start_time) / 1000
            update_instruction_surface(instruction_surface, brush_size, algorithm_time, path_length)
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.put(neighbor)
                neighbor.make_open()

        draw()
        win.blit(instruction_surface, (SCREEN_SIZE, 0))
        pygame.display.update()

        if current != start and current != end:
            current.make_closed()

    return False

def greedy_best_first(draw, grid, start, end, win, instruction_surface, brush_size):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}

    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0

    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    start_time = pygame.time.get_ticks()

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            path_length = reconstruct_path(came_from, end, draw,start,end)
            end_time = pygame.time.get_ticks()
            algorithm_time = (end_time - start_time) / 1000
            update_instruction_surface(instruction_surface, brush_size, algorithm_time,path_length)
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()
        win.blit(instruction_surface, (SCREEN_SIZE, 0))
        pygame.display.update()

        if current != start and current != end:
            current.make_closed()
    return False

def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)
    return grid

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GRID_COLOR, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GRID_COLOR, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
    win.fill(BACKGROUND_COLOR)
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win, rows, width)

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    x,y = pos
    row = y // gap
    col = x // gap
    return row, col

def get_grid_size():
    size = 50  # Default size
    root = Tk()
    root.title("Grid Size")
    root.geometry("300x100")

    def on_submit():
        nonlocal size
        try:
            input_size = int(entry.get())
            if 10 <= input_size <= 200:
                size = input_size
                root.quit()
            else:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a number between 10 and 100.")

    label = Label(root, text="Enter grid size (10-200):")
    label.pack(pady=10)
    entry = Entry(root)
    entry.pack()
    button = Button(root, text="Submit", command=on_submit)
    button.pack(pady=10)
    root.mainloop()
    root.destroy()
    return size

def draw_brush(grid, row, col, brush_size, action):
    for i in range(-brush_size + 1, brush_size):
        for j in range(-brush_size + 1, brush_size):
            if 0 <= row + i < len(grid) and 0 <= col + j < len(grid):
                spot = grid[row + i][col + j]
                if action == "obstacle":
                    spot.make_obstacle()
                elif action == "erase":
                    spot.reset()

def create_instruction_surface(current_algorithm):
    instruction_surface = pygame.Surface((SIDE_PANEL_WIDTH, SCREEN_SIZE))
    instruction_surface.fill(BACKGROUND_COLOR)

    title_font = pygame.font.SysFont('arial', 24, bold=True)
    subtitle_font = pygame.font.SysFont('arial', 18, bold=True)
    text_font = pygame.font.SysFont('arial', 16)

    # Title
    title = title_font.render("Pathfinding Visualizer", True, PANEL_TITLE_COLOR)
    instruction_surface.blit(title, (10, 10))

    # Current Algorithm
    algo_text = subtitle_font.render(f"Current Algorithm: {current_algorithm}", True, PANEL_SUBTITLE_COLOR)
    instruction_surface.blit(algo_text, (10, 40))

    # Instructions
    instructions = [
        ("Controls:", subtitle_font, PANEL_SUBTITLE_COLOR),
        ("Left Click: Place start, end, and obstacles", text_font, TEXT_COLOR),
        ("Right Click: Remove nodes", text_font, TEXT_COLOR),
        ("Space: Start algorithm", text_font, TEXT_COLOR),
        ("C: Clear grid", text_font, TEXT_COLOR),
        ("Up Arrow: Increase brush size", text_font, TEXT_COLOR),
        ("Down Arrow: Decrease brush size", text_font, TEXT_COLOR),
        ("T: Toggle algorithm", text_font, TEXT_COLOR),
        ("V: Toggle real-time visualization", text_font, TEXT_COLOR),
        ("", text_font, TEXT_COLOR),
        ("Colors:", subtitle_font, PANEL_SUBTITLE_COLOR),
        ("Green: Start node", text_font, START_COLOR),
        ("Red: End node", text_font, END_COLOR),
        ("Black: Obstacles", text_font, OBSTACLE_COLOR),
        ("Blue: Path", text_font, PATH_COLOR),
        ("Cyan: Open set", text_font, OPEN_COLOR),
        ("Orange: Closed set", text_font, CLOSED_COLOR)
    ]

    y_offset = 70
    for line, font, color in instructions:
        text = font.render(line, True, color)
        instruction_surface.blit(text, (10, y_offset))
        y_offset += 25

    return instruction_surface

def update_instruction_surface(surface, brush_size, algorithm_time=None,path_length=None):
    # Clear the bottom part of the surface
    pygame.draw.rect(surface, BACKGROUND_COLOR, (0, 500, SIDE_PANEL_WIDTH, 300))

    font = pygame.font.SysFont('arial', 16)

    # Brush size
    brush_text = font.render(f"Current Brush Size: {brush_size}", True, TEXT_COLOR)
    surface.blit(brush_text, (10, 500))

    # Algorithm time
    if algorithm_time is not None:
        time_text = font.render(f"Algorithm Time: {algorithm_time:.2f} seconds", True, TEXT_COLOR)
        surface.blit(time_text, (10, 530))
        # Path length
    if path_length is not None:
        path_text = font.render(f"Path Length: {path_length} nodes", True, TEXT_COLOR)
        surface.blit(path_text, (10, 560))

def clear_path(grid):
    for row in grid:
        for spot in row:
            if not spot.is_obstacle() and not spot.is_start() and not spot.is_end():
                spot.reset()

def main(win, width):
    ROWS = get_grid_size()
    grid = make_grid(ROWS, width)

    start = None
    end = None
    brush_size = 1
    real_time = True

    algorithms = ["A*", "Dijkstra", "BFS", "DFS", "Greedy Best-First"]
    current_algorithm = 0

    instruction_surface = create_instruction_surface(algorithms[current_algorithm])
    update_instruction_surface(instruction_surface, brush_size)

    run = True

    # Initial full draw
    draw(win, grid, ROWS, width)
    win.blit(instruction_surface, (SCREEN_SIZE, 0))
    pygame.display.update()

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:  # LEFT
                pos = pygame.mouse.get_pos()
                if pos[0] < SCREEN_SIZE:  # Ensure click is within the grid
                    row, col = get_clicked_pos(pos, ROWS, width)
                    spot = grid[row][col]
                    if not start and spot != end:
                        start = spot
                        start.make_start()
                    elif not end and spot != start:
                        end = spot
                        end.make_end()
                    elif spot != end and spot != start:
                        draw_brush(grid, row, col, brush_size, "obstacle")
                    draw(win, grid, ROWS, width)
                    win.blit(instruction_surface, (SCREEN_SIZE, 0))
                    pygame.display.update()

            elif pygame.mouse.get_pressed()[2]:  # RIGHT
                pos = pygame.mouse.get_pos()
                if pos[0] < SCREEN_SIZE:  # Ensure click is within the grid
                    row, col = get_clicked_pos(pos, ROWS, width)
                    spot = grid[row][col]
                    if spot == start:
                        start = None
                    elif spot == end:
                        end = None
                    draw_brush(grid, row, col, brush_size, "erase")
                    draw(win, grid, ROWS, width)
                    win.blit(instruction_surface, (SCREEN_SIZE, 0))
                    pygame.display.update()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    algorithm_func = [a_star, dijkstra, bfs, dfs, greedy_best_first][current_algorithm]

                    if real_time:
                        algorithm_func(lambda: draw(win, grid, ROWS, width), grid, start, end, win, instruction_surface, brush_size)
                    else:
                        algorithm_func(lambda: None, grid, start, end, win, instruction_surface, brush_size)
                        draw(win, grid, ROWS, width)
                    win.blit(instruction_surface, (SCREEN_SIZE, 0))
                    pygame.display.update()

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)
                    instruction_surface = create_instruction_surface(algorithms[current_algorithm])
                    update_instruction_surface(instruction_surface, brush_size)
                    draw(win, grid, ROWS, width)
                    win.blit(instruction_surface, (SCREEN_SIZE, 0))
                    pygame.display.update()

                if event.key == pygame.K_UP:
                    brush_size = min(brush_size + 1, 5)
                    update_instruction_surface(instruction_surface, brush_size)
                    win.blit(instruction_surface, (SCREEN_SIZE, 0))
                    pygame.display.update()

                if event.key == pygame.K_DOWN:
                    brush_size = max(brush_size - 1, 1)
                    update_instruction_surface(instruction_surface, brush_size)
                    win.blit(instruction_surface, (SCREEN_SIZE, 0))
                    pygame.display.update()

                if event.key == pygame.K_v:
                    real_time = not real_time

                if event.key == pygame.K_t:
                    current_algorithm = (current_algorithm + 1) % len(algorithms)
                    instruction_surface = create_instruction_surface(algorithms[current_algorithm])
                    update_instruction_surface(instruction_surface, brush_size)
                    clear_path(grid)
                    if start:
                        start.make_start()
                    if end:
                        end.make_end()
                    draw(win, grid, ROWS, width)
                    win.blit(instruction_surface, (SCREEN_SIZE, 0))
                    pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    WIN = pygame.display.set_mode((SCREEN_SIZE + SIDE_PANEL_WIDTH, SCREEN_SIZE))
    pygame.display.set_caption("Pathfinding Algorithm Visualizer")
    main(WIN, SCREEN_SIZE)
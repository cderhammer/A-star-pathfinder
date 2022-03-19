from operator import truediv
import pygame
import math
from queue import PriorityQueue

"""This program simulates the A* path finding algorithm using Manhattan distance.
    When the program is run, the first press on the left mouse button will create 
    a starting node (blue). The second press on the left mouse button will create an end
    node (orange). Subsequent presses on the left mouse button will create 'barriers' (black).
    The right mouse button will erase any nodes/tiles created. Spacebar runs the program
    and 'C' clears the whole board."""

WIDTH = 500
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding")


# gCost = 0  # current shortest distance to get from the start node to node n
# hCost = 0  # gives an estimate of the distance from node n to the end node
# fCost = gCost + hCost

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)
GREY = (128, 128, 128)
TEAL = (64, 224, 208)
ORANGE = (255, 165, 0)


class Tile:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.width = width
        self.total_rows = total_rows
        self.color = WHITE  # available tile
        self.neighbors = []

    def get_pos(self):
        return self.row, self.col

    def have_visited(self):
        return self.color == RED  # have visited the tile

    def can_visit(self):
        return self.color == GREEN  # can visit the tile

    def barrier(self):
        return self.color == BLACK  # barrier tile

    def start(self):
        return self.color == ORANGE  # starting tile

    def end(self):
        return self.color == TEAL  # ending tile

    def reset(self):
        self.color = WHITE  # reset the tile

    def make_visited(self):
        self.color = RED  # make the visited tile red

    def make_open(self):
        self.color = GREEN  # make available tile green

    def make_barrier(self):
        self.color = BLACK  # make barrier tile black

    def make_start(self):
        self.color = ORANGE  # make starting tile orange

    def make_end(self):
        self.color = TEAL  # make ending tile purple

    def make_path(self):
        self.color = PURPLE  # make path purple

    def draw(self, win):
        pygame.draw.rect(
            win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        # if we can move down rows
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].barrier():
            self.neighbors.append(grid[self.row + 1][self.col])

        # if we can move up rows
        if self.row > 0 and not grid[self.row - 1][self.col].barrier():
            self.neighbors.append(grid[self.row - 1][self.col])

        # if we can move right
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].barrier():
            self.neighbors.append(grid[self.row][self.col + 1])

        # if we can move left
        if self.col > 0 and not grid[self.row][self.col - 1].barrier():
            self.neighbors.append(grid[self.row][self.col - 1])


def heuristic(p1, p2):  # defines the heuristic value
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

# creates shortest path from start to end using Manhattan distance
def reconstruct_path(came_from_node, current, draw):
    while current in came_from_node:
        current = came_from_node[current]
        current.make_path()
        draw()


def algorithm(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from_node = {}

    # current shortest distance to get from the start node to node n
    gCost = {tile: float("inf") for row in grid for tile in row}
    gCost[start] = 0
    fCost = {tile: float("inf") for row in grid for tile in row}
    fCost[start] = heuristic(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():  # exit the loop
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]  # start node
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from_node, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_gCost = gCost[current] + 1

            if temp_gCost < gCost[neighbor]:
                came_from_node[neighbor] = current
                gCost[neighbor] = temp_gCost
                fCost[neighbor] = temp_gCost + \
                    heuristic(neighbor.get_pos(), end.get_pos())

                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((fCost[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_visited()

    return False


def make_grid(rows, width):  # make grid
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            tile = Tile(i, j, gap, rows)
            grid[i].append(tile)  # store a list inside of lists in the grid

    return grid


def draw_grid(win, rows, width):  # draw gridlines
    gap = width // rows
    for i in range(rows):
        # creates the black lines that separate the tiles
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win, grid, rows, width):  # draw blank canvas
    win.fill(WHITE)

    for row in grid:
        for tile in row:
            tile.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()


def get_mouse_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col


def main(win, width):
    ROWS = 25
    grid = make_grid(ROWS, width)

    start = None
    end = None

    run = True

    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:  # left mouse button
                pos = pygame.mouse.get_pos()
                row, col = get_mouse_pos(pos, ROWS, width)
                tile = grid[row][col]
                if not start:
                    start = tile
                    start.make_start()

                elif not end and tile != start:
                    end = tile
                    end.make_end()

                elif tile != end and tile != start:
                    tile.make_barrier()

            elif pygame.mouse.get_pressed()[2]:  # right mouse button erases
                pos = pygame.mouse.get_pos()
                row, col = get_mouse_pos(pos, ROWS, width)
                tile = grid[row][col]
                tile.reset()

                if tile == start:
                    start = None

                if tile == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:  # start the program
                    for row in grid:
                        for tile in row:
                            tile.update_neighbors(grid)

                    algorithm(lambda: draw(win, grid, ROWS, width),
                              grid, start, end)

                if event.key == pygame.K_c:  # reset the grid
                    end = None
                    start = None
                    grid = make_grid(ROWS, width)

    pygame.quit()


main(WIN, WIDTH)

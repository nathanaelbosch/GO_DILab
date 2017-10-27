import numpy as np
import pygame
import sys
import time

from src.view import View
from src.view import Move
from src.model.Game import BLACK
from src.model.Game import WHITE

brown = (165, 42, 42)
black = (0, 0, 0)
white = (255, 255, 255)
size = (500, 500)
board_size = 400
offset = 50
stone_radius = 20


class PygameGuiView(View):

    def __init__(self, game):
        View.__init__(self, game)
        self.running = False
        self.cell_size = board_size / (self.game.size - 1)

    def open(self, game_controller):
        pygame.init()
        self.running = True
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption('Go')
        self.render()

        self.is_ready = True

        while self.running:
            event = pygame.event.poll()
            if event.type == pygame.MOUSEBUTTONUP:
                x, y = event.pos  # TODO this is super unreliable on macOS but seems fine on Windows?!
                col = int(round((x - offset) / self.cell_size))
                row = int(round((y - offset) / self.cell_size))
                game_controller.current_player.receive_next_move_from_gui(Move(col, row))
            if event.type == pygame.QUIT:
                self.running = False

        pygame.quit()
        sys.exit(0)  # kinda brutal, I failed to close the pygame window properly though

    def show_player_turn_start(self, name):
        pass

    def show_player_turn_end(self, name):
        self.render()

    def render(self):
        # board
        self.screen.fill(brown)
        for i in range(0, self.game.size):
            # horizontals
            pygame.draw.line(self.screen, black, [offset, offset + i * self.cell_size],
                             [offset + board_size, offset + i * self.cell_size], 1)
            # verticals
            pygame.draw.line(self.screen, black, [offset + i * self.cell_size, offset],
                             [offset + i * self.cell_size, offset + board_size], 1)
        # stones
        b = self.game.board.copy()  # is it necessary to copy it?
        b[b == BLACK] = 2
        b[b == WHITE] = 3
        black_rows, black_cols = np.where(b == 2)
        white_rows, white_cols = np.where(b == 3)
        for i in range(0, len(black_rows)):
            indices = black_cols[i], black_rows[i]
            self.draw_stone(indices, black)
        for i in range(0, len(white_rows)):
            indices = white_cols[i], white_rows[i]
            self.draw_stone(indices, white)

        pygame.display.flip()  # update the screen

    def draw_stone(self, indices, col):
        pos = (int(offset + self.cell_size * indices[0]), int(offset + indices[1] * self.cell_size))
        # make the circles look nicer -> antialiasing, see stackoverflow.com/a/26774279/2474159 TODO
        pygame.draw.circle(self.screen, col, pos, stone_radius, 0)

    def show_error(self, msg):
        pass

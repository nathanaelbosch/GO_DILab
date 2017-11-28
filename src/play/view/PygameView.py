import numpy as np
import pygame
from pygame import gfxdraw
import sys

from src.play.model.Board import BLACK, WHITE
from src.play.model.Move import Move

brown = (165, 42, 42)
black = (0, 0, 0)
white = (255, 255, 255)
yellow = (255, 255, 0)
orange = (255, 165, 0)
blue = (0, 0, 255)

window_size = (500, 600)
board_top_left_coord = (50, 100)  # coordinate of the boards top left point
board_size = 400
stone_radius = 20


class PygameView:

    def __init__(self, controller):
        controller.view = self
        self.controller = controller
        self.game = controller.game
        self.running = False
        self.cell_size = board_size / (self.game.size - 1)
        self.buttons = []
        self.labels = []

    def open(self):
        pygame.init()
        self.running = True
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption('Go')
        self.buttons.append(Button(
            210, 530, 80, 40, 'Pass', self.screen, self.send_pass_move))
        self.labels.append(Label(
            100, 30, 300, 40, self.get_turn_label_text, self.screen))
        self.render()

        while self.running:
            event = pygame.event.poll()
            if event.type == pygame.MOUSEBUTTONUP:
                x, y = event.pos
                col = int(round((x - board_top_left_coord[0]) /
                                self.cell_size))
                row = int(round((y - board_top_left_coord[1]) /
                                self.cell_size))
                if 0 <= col < self.game.size and 0 <= row < self.game.size:
                    self.controller.receive_move_from_gui(Move(col, row))
                for btn in self.buttons:
                    btn.check_mouse_released()
            if event.type == pygame.QUIT:
                self.running = False
            for btn in self.buttons:
                btn.is_mouse_over_btn()
            self.render()

        pygame.quit()
        sys.exit(0)

    def game_ended(self):
        self.labels[0] = (Label(100, 30, 300, 40, lambda: "Game has ended", self.screen))

    def send_pass_move(self):
        self.controller.receive_move_from_gui(Move(is_pass=True))

    def get_turn_label_text(self):
        return self.controller.current_player.name + '\'s (' \
               + self.controller.current_player.color + ') turn'

    def render(self):
        # board
        self.screen.fill(brown)
        for i in range(0, self.game.size):
            # horizontals
            pygame.draw.line(
                self.screen, black,
                [board_top_left_coord[0],
                 board_top_left_coord[1] + i * self.cell_size],
                [board_top_left_coord[0] + board_size,
                 board_top_left_coord[1] + i * self.cell_size], 1)
            # verticals
            pygame.draw.line(
                self.screen, black,
                [board_top_left_coord[0] + i * self.cell_size,
                 board_top_left_coord[1]],
                [board_top_left_coord[0] + i * self.cell_size,
                 board_top_left_coord[1] + board_size], 1)
        # stones
        black_rows, black_cols = np.where(self.game.board == BLACK)
        white_rows, white_cols = np.where(self.game.board == WHITE)
        for i in range(0, len(black_rows)):
            indices = black_cols[i], black_rows[i]
            self.draw_stone(indices, black)
        for i in range(0, len(white_rows)):
            indices = white_cols[i], white_rows[i]
            self.draw_stone(indices, white)
        for btn in self.buttons:
            btn.draw()
        for label in self.labels:
            label.draw()
        pygame.display.flip()  # update the screen

    def draw_stone(self, indices, col):
        x = int(board_top_left_coord[0] + self.cell_size * indices[0])
        y = int(board_top_left_coord[1] + indices[1] * self.cell_size)
        # antialiasing via stackoverflow.com/a/26774279/2474159
        gfxdraw.aacircle(self.screen, x, y, stone_radius, col)
        gfxdraw.filled_circle(self.screen, x, y, stone_radius, col)


class Button:
    """Adapted from
    gamedev.net/forums/topic/686666-pygame-buttons/?do=findComment&comment=5333411
    """
    def __init__(self, x, y, w, h, text, screen, on_click):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.rect = (self.x, self.y, self.w, self.h)
        pygame.font.init()
        self.font = pygame.font.Font(None, 25)  # compute font size from h?
        self.text = text
        self.mouse_is_over_btn = False
        self.screen = screen
        self.on_click = on_click

    def draw(self):
        if self.mouse_is_over_btn:
            bg = yellow
        else:
            bg = orange
        surf = self.font.render(self.text, True, black, bg)
        xo = self.x + (self.w - surf.get_width()) / 2
        yo = self.y + (self.h - surf.get_height()) / 2
        self.screen.fill(bg, self.rect)
        pygame.draw.rect(self.screen, black, self.rect, 1)
        self.screen.blit(surf, (xo, yo))

    def is_mouse_over_btn(self):
        pos = pygame.mouse.get_pos()
        self.mouse_is_over_btn = (
            self.x <= pos[0] < self.x + self.w and
            self.y <= pos[1] < self.y + self.h)

    def check_mouse_released(self):
        if self.mouse_is_over_btn:
            self.on_click()


class Label:  # make Button extend Label?

    def __init__(self, x, y, w, h, get_text, screen):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.rect = (self.x, self.y, self.w, self.h)
        pygame.font.init()
        self.font = pygame.font.Font(None, 30)
        self.get_text = get_text  # method, get's called in draw()
        self.screen = screen

    def draw(self):
        surf = self.font.render(self.get_text(), True, yellow, brown)
        xo = self.x + (self.w - surf.get_width()) / 2
        yo = self.y + (self.h - surf.get_height()) / 2
        self.screen.blit(surf, (xo, yo))

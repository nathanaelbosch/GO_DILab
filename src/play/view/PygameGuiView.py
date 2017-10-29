import numpy as np
import pygame
from pygame import gfxdraw
import sys
import time
from src.play import ConsoleView
from src.play.view import View
from src.play.view import Move
from src.play.model.Game import BLACK
from src.play.model.Game import WHITE

brown = (165, 42, 42)
black = (0, 0, 0)
white = (255, 255, 255)
yellow = (255, 255, 0)
orange = (255, 165, 0)
blue = (0, 0, 255)
size = (500, 600)
board_size = 400
x_offset = 50
y_offset = 100
stone_radius = 20


class PygameGuiView(View):

    def __init__(self, game):
        View.__init__(self, game)
        self.console_view = ConsoleView(game)
        self.console_view.print_board()
        self.running = False
        self.cell_size = board_size / (self.game.size - 1)
        self.buttons = []
        self.labels = []

    def open(self, game_controller):
        self.game_controller = game_controller
        pygame.init()
        self.running = True
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption('Go')
        self.buttons.append(Button(210, 530, 80, 40, 'Pass', self.screen, self.send_pass_move))
        self.labels.append(Label(100, 30, 300, 40, self.get_turn_label_text, self.screen))
        self.render()

        while self.running:
            event = pygame.event.poll()
            if event.type == pygame.MOUSEBUTTONUP:
                x, y = event.pos
                col = int(round((x - x_offset) / self.cell_size))
                row = int(round((y - y_offset) / self.cell_size))
                if 0 <= col < self.game.size and 0 <= row < self.game.size:
                    self.game_controller.current_player.receive_next_move_from_gui(Move(col, row))
                for btn in self.buttons:
                    btn.check_mouse_released()
            if event.type == pygame.QUIT:
                self.running = False
            for btn in self.buttons:
                btn.is_mouse_over_btn()
            time.sleep(0.025)  # this avoids the brief flashing of stones, not a good solution though TODO
            self.render()

        # this exiting mechanism doesn't work (on macOS at least), causes a freeze TODO
        pygame.quit()
        sys.exit(0)

    def show_player_turn_start(self, name):
        self.console_view.show_player_turn_start(name)

    def show_player_turn_end(self, name):
        self.console_view.show_player_turn_end(name)

    def send_pass_move(self):
        self.game_controller.current_player.receive_next_move_from_gui(Move(is_pass=True))

    def get_turn_label_text(self):
        return 'It\'s ' + self.game_controller.current_player.name + '\'s turn'

    def render(self):
        # board
        self.screen.fill(brown)
        for i in range(0, self.game.size):
            # horizontals
            pygame.draw.line(self.screen, black, [x_offset, y_offset + i * self.cell_size],
                             [x_offset + board_size, y_offset + i * self.cell_size], 1)
            # verticals
            pygame.draw.line(self.screen, black, [x_offset + i * self.cell_size, y_offset],
                             [x_offset + i * self.cell_size, y_offset + board_size], 1)
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
        for btn in self.buttons:
            btn.draw()
        for label in self.labels:
            label.draw()
        pygame.display.flip()  # update the screen

    def draw_stone(self, indices, col):
        x = int(x_offset + self.cell_size * indices[0])
        y = int(y_offset + indices[1] * self.cell_size)
        # antialiasing via stackoverflow.com/a/26774279/2474159
        pygame.gfxdraw.aacircle(self.screen, x, y, stone_radius, col)
        pygame.gfxdraw.filled_circle(self.screen, x, y, stone_radius, col)

    def show_error(self, msg):
        self.console_view.show_error(msg)


class Button:  # adapted from gamedev.net/forums/topic/686666-pygame-buttons/?do=findComment&comment=5333411

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
        self.mouse_is_over_btn = self.x <= pos[0] < self.x + self.w and self.y <= pos[1] < self.y + self.h

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

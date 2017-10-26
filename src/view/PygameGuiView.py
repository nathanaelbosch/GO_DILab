import numpy as np
import pygame
import sys
from src.view import View
from src.model.Game import BLACK
from src.model.Game import WHITE

brown = (165, 42, 42)
black = (0, 0, 0)
white = (255, 255, 255)
size = (500, 500)
width = 20
height = 20


class PygameGuiView(View):

    def __init__(self, game):
        View.__init__(self, game)
        self.running = False

    def open(self, game_controller):
        pygame.init()
        self.running = True
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption('Go')
        self.draw_board()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    print(event)
                if event.type == pygame.QUIT:
                    self.running = False
            pygame.display.flip()

        pygame.quit()
        sys.exit(0)  # kinda brutal, I failed to close the pygame window properly though

    def show_player_turn_start(self, name):
        pass  # TODO

    def show_player_turn_end(self, name):
        self.draw_board()
        b = self.game.board.copy()
        b[b == BLACK] = 2
        b[b == WHITE] = 3
        black_rows, black_cols = np.where(b == 2)
        white_rows, white_cols = np.where(b == 3)
        # This method does not really make sense because it draws circles over previously existing ones.
        # Will update once I get the hang of the code.
        # make the circles look nicer -> antialiasing, see stackoverflow.com/a/26774279/2474159 TODO
        for i in range(0, len(black_rows)):
            pygame.draw.circle(self.screen, (0, 0, 0), (50 + 50 * black_cols[i], 50 + black_rows[i] * 50), 20, 0)
        for i in range(0, len(white_rows)):
            pygame.draw.circle(self.screen, (255, 255, 255), (50 + 50 * white_cols[i], 50 + 50 * white_rows[i]), 20, 0)

    def show_error(self, msg):
        pass

    def draw_board(self):
        self.screen.fill(brown)
        for i in range(1, 10):
            pygame.draw.line(self.screen, black, [50, i * 50], [450, i * 50], 1)
            pygame.draw.line(self.screen, black, [50 * i, 50], [50 * i, 450], 1)

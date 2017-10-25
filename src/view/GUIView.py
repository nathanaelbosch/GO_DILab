from src.view import View
import numpy as np
from src.model.Game import BLACK
from src.model.Game import WHITE

import pygame


class GUIView(View):
    def __init__(self, game):
        View.__init__(self, game)
        pygame.init()
        black = (0, 0, 0)
        white = (255, 255, 255)
        size = (550, 550)
        brown = (165, 42, 42)
        width = 20
        height = 20
        self.screen = pygame.display.set_mode(size)
        self.screen.fill(brown)
        for i in range(1, 10):
            pygame.draw.line(self.screen, black, [50, i * 50], [450, i * 50], 1)
            pygame.draw.line(self.screen, black, [50 * i, 50], [50 * i, 450], 1)
        pygame.display.set_caption('My First Game')
        pygame.display.flip()

    def show_player_turn_start(self, name):
        # Once GUI is completely implemented, this will become unnecessary
        print('Its player ' + name + '\'s turn. Submit your desired location...')

    def show_player_turn_end(self, name):
        b = self.game.board.copy()
        b[b == BLACK] = 2
        b[b == WHITE] = 3
        black_rows, black_cols = np.where(b == 2)
        white_rows, white_cols = np.where(b==3)
        #This method does not really make sense because it draws circles over previously existing ones.
        # Will update once I get the hang of the code.
        for i in range(0, len(black_rows)):
            pygame.draw.circle(self.screen, (0, 0, 0),( 50 + 50*black_cols[i],50 + black_rows[i]*50),20, 0)
        for i in range(0,len(white_rows)):
            pygame.draw.circle(self.screen,(255,255,255),(50 + 50*white_cols[i],50 + 50*white_rows[i]),20,0)
        pygame.display.flip()


    def show_error(self, msg):
        pass

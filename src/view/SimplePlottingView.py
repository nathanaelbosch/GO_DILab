from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import tkinter  # Python's standard GUI

from src.view import View
from src.view import Move
from src.model.Game import BLACK
from src.model.Game import WHITE

GAME_CONTROLLER_START_DELAY = 500  # ms


class SimplePlottingView(View):

    def __init__(self, game):
        View.__init__(self, game)
        self.root = tkinter.Tk()

    def open(self, game_controller):
        self.game_controller = game_controller
        self.root.wm_title("Go")

        n = self.game.size
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()
        ax.set(title='Go')
        ax.set_facecolor('y')

        for i in range(0, n):
            for j in range(0, n):
                col = 'g'
                if self.game.board.item(i, j) == BLACK:
                    col = 'b'
                if self.game.board.item(i, j) == WHITE:
                    col = 'w'
                ax.add_artist(plt.Circle((i, j), 0.4, color=col))

        ax.set_xlim([-1, n])
        ax.set_ylim([-1, n])
        plt.xticks(np.arange(0, n, 1))
        plt.yticks(np.arange(0, n, 1))
        plt.grid()

        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.show()
        canvas.get_tk_widget().pack()

        pass_btn = tkinter.Button(master=self.root, text='Pass', command=self.pass_btn_clicked)
        pass_btn.pack(side=tkinter.BOTTOM)

        canvas.mpl_connect('button_release_event', self.handle_mouse_event)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.root.after(GAME_CONTROLLER_START_DELAY, game_controller.start)
        self.root.mainloop()

    def handle_mouse_event(self, event):
        # accept only left-clicks in canvas area
        if (event.inaxes is None) or (event.button != 1):
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if 0 <= col < self.game.size and 0 <= row < self.game.size:
            self.game_controller.current_player.receive_next_move_from_gui(Move(row, col))

    def quit(self):
        self.root.quit()
        self.root.destroy()

    def pass_btn_clicked(self):
        self.game_controller.current_player.receive_gui_event(Move(is_pass=True))

    def show_player_turn_start(self, name):
        pass

    def show_player_turn_end(self, name):
        pass

    def show_error(self, msg):
        pass

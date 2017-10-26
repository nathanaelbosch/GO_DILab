
class Move:

    def __init__(self, col=None, row=None, is_pass=True):
        self.col = col
        self.row = row
        self.is_pass = is_pass

    def __repr__(self):
        return '(col=' + str(self.col) + ' / row=' + str(self.row) + ')'

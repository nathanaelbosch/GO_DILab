
class Move:

    def __init__(self, row=None, col=None, is_pass=True):
        self.row = row
        self.col = col
        self.is_pass = is_pass

    def __repr__(self):
        return '(row=' + str(self.row) + ' / col=' + str(self.col) + ')'

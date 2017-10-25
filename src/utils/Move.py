
class Move:

    def __init__(self, col=-1, row=-1, is_pass=True):
        self.col = col
        self.row = row
        self.is_pass = is_pass

    def __repr__(self):
        return '(' + str(self.col) + ' / ' + str(self.row) + ')'

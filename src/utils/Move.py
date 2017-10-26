
class Move:

    def __init__(self, col=None, row=None, is_pass=False):
        self.col = col
        self.row = row
        self.is_pass = is_pass

    def get_loc_as_matrix_coords(self):
        return self.row, self.col  # row/col instead of col/row

    def __repr__(self):
        if self.is_pass:
            return 'pass'
        return '(' + str(self.col) + '/' + str(self.row) + ')'

"""Class that handles moves!

Supports alternative constructors using `from_*`, so that it can be
constructed from the sgf format, gtp, and matrix locations.
Same goes for output, using the `to_*` functions
"""
import math


class Move:
    def __init__(self, col=None, row=None, is_pass=False):
        self.col = col
        self.row = row
        self.is_pass = is_pass

    @classmethod
    def from_sgf(cls, string):
        """Create instance from string using the SGF standard

        Examples
        --------
        >>> print(Move.from_sgf('ef'))
        (4/5)
        >>> print(Move.from_sgf('AA'))
        (0/0)
        >>> print(Move.from_sgf(''))
        pass
        """
        string = string.lower()
        if string == '':
            return cls(is_pass=True)
        else:
            col = ord(string[0]) - ord('a')
            row = ord(string[1]) - ord('a')
            return cls(col, row)

    @classmethod
    def from_gtp(cls, string, size=9):
        """Create instance from string using the SGF standard

        Examples
        --------
        >>> print(Move.from_gtp('E6'))
        (4/5)
        >>> print(Move.from_gtp('pass'))
        pass
        """
        string = string.lower()
        if string == 'pass':
            return cls(is_pass=True)
        else:
            _ord = ord(string[0])
            # i is excluded from board coordinates in GTP
            if _ord >= ord('j'):
                _ord -= 1
            col = _ord - ord('a')
            row = size - int(string[1])
            # raise possible parsing errors here TODO
            return cls(col, row)

    @classmethod
    def from_matrix_location(cls, loc):
        """Create instance from (int, int)

        Examples
        --------
        >>> print(Move.from_matrix_location((1,3)))
        (3/1)
        """
        return cls(loc[1], loc[0])

    def to_sgf(self):
        """Output move following the SGF standard

        Examples
        --------
        >>> Move(0,1).to_sgf()
        'ab'
        >>> Move(is_pass=True).to_sgf()
        ''
        """
        if self.is_pass:
            return ''
        else:
            col = chr(self.col + ord('a'))
            row = chr(self.row + ord('a'))
            return col+row

    def to_gtp(self, size=9):
        """Output move following the GTP standard

        Examples
        --------
        >>> Move(0,1).to_gtp()
        'A2'
        >>> Move(is_pass=True).to_gtp()
        'pass'
        """
        if self.is_pass:
            return 'pass'
        else:
            _chr = self.col + ord('A')
            # i is excluded from board coordinates in GTP
            if self.col >= 8:
                _chr += 1
            col = str(chr(_chr))
            row = str(size - self.row)
            return col + row

    def to_matrix_location(self):
        return self.row, self.col  # row/col instead of col/row

    def __str__(self):
        if self.is_pass:
            return 'pass'
        return '(' + str(self.col) + '/' + str(self.row) + ')'

    # TODO
    # it would be better style if this would throw an InvalidMove_Error, that creates
    # circular import-errors though, the error class would have to be moved out of Game
    def is_on_board(self, size=9):
        return self.is_pass is True or 0 <= self.col < size and 0 <= self.row < size

    def to_flat_idx(self, size=9):
        """

        Used by NNBot to get the index in a matrix that got serialized into
        a single array in the following way:
        AB
        CD
        -> ABCD
        """
        return self.row * size + self.col

    @classmethod
    def from_flat_idx(cls, flat_move, size=9):
        """

        Used by NNBot to get the index in a matrix that got serialized into
        a single array in the following way:
        AB
        CD
        -> ABCD
        """
        row = int(math.floor(flat_move / 9))
        col = int(flat_move % 9)
        return cls(col=col, row=row)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

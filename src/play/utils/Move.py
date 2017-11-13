"""Class that handles moves!

Supports alternative constructors using `from_*`, so that it can be
constructed from the sgf format, gtp, and matrix locations.
Same goes for output, using the `to_*` functions
"""


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
    def from_gtp(cls, string):
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
            col = ord(string[0]) - ord('a')
            row = int(string[1]) - 1
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

    def to_gtp(self):
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
            col = str(chr(self.col + ord('A')))
            row = str(self.row + 1)
            return col+row

    def to_matrix_location(self):
        return self.row, self.col  # row/col instead of col/row

    def __str__(self):
        if self.is_pass:
            return 'pass'
        return '(' + str(self.col) + '/' + str(self.row) + ')'

    # TODO
    # it would be better style if this would throw an InvalidMove_Error, that creates
    # circular import-errors though, the error class would have to be moved out of Game
    def is_on_board(self, size):
        return 0 < self.col < size and 0 < self.row < size


if __name__ == '__main__':
    import doctest
    doctest.testmod()

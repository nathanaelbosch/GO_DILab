"""Class that handles moves!

We want to create moves out of matrix locations, sgf strings, gtp strings, ...
We also want to output the move in different formats!
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

    def to_matrix_location(self):
        return self.row, self.col  # row/col instead of col/row

    def __str__(self):
        if self.is_pass:
            return 'pass'
        return '(' + str(self.col) + '/' + str(self.row) + ')'


if __name__ == '__main__':
    import doctest
    doctest.testmod()

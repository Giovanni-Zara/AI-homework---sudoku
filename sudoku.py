import numpy as np
from typing import List, Tuple, Set, Optional
from copy import deepcopy

class SudokuState:
    """Represents a Sudoku puzzle state."""
    
    def __init__(self, grid: np.ndarray):
        """
        Initialize Sudoku state.
        Args:
            grid: 9x9 numpy array where 0 represents empty cells
        """
        self.grid = grid.copy()  # gotta copy or it breaks everything
        self.size = 9
        self.box_size = 3
    
    def is_valid_move(self, row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid."""
        # Check row
        if num in self.grid[row, :]:  # cant have dupes in same row obv
            return False
        
        # Check column
        if num in self.grid[:, col]:  # same for cols
            return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)  # math trick to find which box we're in
        if num in self.grid[box_row:box_row+3, box_col:box_col+3]:
            return False
        
        return True
    
    def get_possible_values(self, row: int, col: int) -> Set[int]:
        """Get all possible values for a cell."""
        if self.grid[row, col] != 0:  # if already filled return empty set duh
            return set()
        
        possible = set(range(1, 10))  # start w/ all nums 1-9
        
        # Remove row values
        possible -= set(self.grid[row, :])  # kick out anything already in row
        
        # Remove column values
        possible -= set(self.grid[:, col])  # kick out col stuff too
        
        # Remove box values
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        box_values = self.grid[box_row:box_row+3, box_col:box_col+3].flatten()
        possible -= set(box_values)  # bye bye box values
        
        return possible  # whats left r the goods
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Return list of empty cell coordinates."""
        return [(i, j) for i in range(self.size) 
                for j in range(self.size) if self.grid[i, j] == 0]  # 0 is empty reminder
    
    def is_complete(self) -> bool:
        """Check if puzzle is completely filled."""
        return 0 not in self.grid  # no zeros = done
    
    def is_valid(self) -> bool:
        """Check if current state is valid (no conflicts)."""
        # Check rows
        for i in range(self.size):
            row = self.grid[i, :]
            non_zero = row[row != 0]  # ignore empties
            if len(non_zero) != len(set(non_zero)):  # if len diff = dupes exist -> bad
                return False
        
        # Check columns
        for j in range(self.size):
            col = self.grid[:, j]
            non_zero = col[col != 0]
            if len(non_zero) != len(set(non_zero)):  # same check 4 cols
                return False
        
        # Check boxes
        for box_i in range(3):
            for box_j in range(3):
                box = self.grid[box_i*3:(box_i+1)*3, box_j*3:(box_j+1)*3].flatten()
                non_zero = box[box != 0]
                if len(non_zero) != len(set(non_zero)):  # check all 9 boxes
                    return False
        
        return True
    
    def is_goal(self) -> bool:
        """Check if state is a valid solution."""
        return self.is_complete() and self.is_valid()  # must be both full AND correct
    
    def copy(self) -> 'SudokuState':
        """Create a deep copy of the state."""
        return SudokuState(self.grid.copy())  # deep copy like before
    
    def place_number(self, row: int, col: int, num: int) -> 'SudokuState':
        """Return new state with number placed at position."""
        new_state = self.copy()  # always make new state, dont modify original!!!
        new_state.grid[row, col] = num
        return new_state
    
    def get_successors(self) -> List[Tuple['SudokuState', Tuple[int, int, int]]]:
        """
        Generate successor states.
        Returns list of (state, action) tuples where action is (row, col, num).
        """
        empty_cells = self.get_empty_cells()
        if not empty_cells:  # no empty = no successors
            return []
        
        # Choose cell with minimum remaining values (MRV heuristic)
        best_cell = min(empty_cells, 
                       key=lambda cell: len(self.get_possible_values(cell[0], cell[1])))  # pick cell w least options 
        row, col = best_cell
        
        successors = []
        for num in self.get_possible_values(row, col):  # try all valid nums
            new_state = self.place_number(row, col, num)
            successors.append((new_state, (row, col, num)))
        
        return successors
    
    def heuristic(self) -> int:
        """
        Heuristic function for A*: number of empty cells.
        Lower is better (closer to goal).
        """
        return np.count_nonzero(self.grid == 0)  # less empties = closer to goal 
    
    def __hash__(self):
        return hash(self.grid.tobytes())  # needed 4 sets/dicts
    
    def __eq__(self, other):
        return isinstance(other, SudokuState) and np.array_equal(self.grid, other.grid)  # compare grids
    
    def __str__(self):
        result = []
        for i in range(self.size):
            if i % 3 == 0 and i != 0:
                result.append("-" * 21)  # horizontal lines every 3 rows
            row = []
            for j in range(self.size):
                if j % 3 == 0 and j != 0:
                    row.append("| ")  # vertical lines every 3 cols
                row.append(str(self.grid[i, j]) if self.grid[i, j] != 0 else ".")
                row.append(" ")
            result.append("".join(row))
        return "\n".join(result)  # make it look pretty


def load_sudoku_from_string(puzzle_string: str) -> SudokuState:
    """
    Load Sudoku from string representation.
    Args:
        puzzle_string: 81-character string where 0 or . represents empty cells
    """
    puzzle_string = puzzle_string.replace('.', '0')  # dots -> zeros
    grid = np.array([int(c) for c in puzzle_string if c.isdigit()]).reshape(9, 9)  # turn str into 9x9 array
    return SudokuState(grid)


'''def load_sudoku_from_file(filename: str) -> SudokuState:             #dunno if needed, for now using strings as examples
    """Load Sudoku from file (9 lines of 9 digits each)."""
    with open(filename, 'r') as f:
        lines = [line.strip().replace('.', '0') for line in f.readlines() if line.strip()]
        grid = np.array([[int(c) for c in line if c.isdigit()] for line in lines[:9]])
    return SudokuState(grid)'''


# Example puzzles
EASY_PUZZLE = """
530070000
600195000
098000060
800060003
400803001
700020006
060000280
000419005
000080079
"""

MEDIUM_PUZZLE = """
003020600
900305001
001806400
008102900
700000008
006708200
002609500
800203009
005010300
"""

HARD_PUZZLE = """
800000000
003600000
070090200
050007000
000045700
000100030
001000068
008500010
090000400
"""


if __name__ == "__main__":
    # Test the implementation
    puzzle = load_sudoku_from_string(EASY_PUZZLE.replace('\n', ''))  # load easy one 4 testing
    print("Initial Sudoku:")
    print(puzzle)
    print(f"\nEmpty cells: {len(puzzle.get_empty_cells())}")
    print(f"Is valid: {puzzle.is_valid()}")
    print(f"Is complete: {puzzle.is_complete()}")
    
    # Test possible values
    empty_cells = puzzle.get_empty_cells()
    if empty_cells:
        row, col = empty_cells[0]
        print(f"\nPossible values for cell ({row}, {col}): {puzzle.get_possible_values(row, col)}")  # check what we can put in first empty
    
    # Test successors
    successors = puzzle.get_successors()
    print(f"\nNumber of successors from initial state: {len(successors)}")  # how many moves possible

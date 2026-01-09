"""
Sudoku Solver using Constraint Satisfaction Problem (CSP) approach
Uses python-constraint library as the CSP solver
"""

import time
from typing import List, Tuple, Optional, Dict
from constraint import Problem, AllDifferentConstraint  # this lib does the heavy lifting, standard csp stuff
import numpy as np

# Import Sudoku representation
from sudoku import SudokuState, load_sudoku_from_string, EASY_PUZZLE, MEDIUM_PUZZLE, HARD_PUZZLE


class SudokuCSPSolver:
    """
    Solves Sudoku by reducing it to CSP and using python-constraint solver.
    
    CSP formulation:
    - Variables: cell_(r,c) for each cell, 81 total
    - Domains: {1..9} for empty cells, {fixed_value} for pre-filled
    - Constraints: AllDifferent for each row, column, and 3x3 box
    """
    
    def __init__(self):
        self.stats = {  # gonna track some stats 4 the report
            'modeling_time': 0,
            'solving_time': 0,
            'total_time': 0,
            'variables': 0,
            'constraints': 0
        }
    
    def _create_csp_from_sudoku(self, sudoku_state: SudokuState) -> Problem:
        """
        Generate CSP problem from SudokuState.
        This is the reduction step: Sudoku -> CSP
        """
        problem = Problem()  # fresh csp problem
        
        grid = sudoku_state.grid
        size = 9
        
        # Step 1: Add variables with their domains
        for row in range(size):
            for col in range(size):
                var_name = f"cell_{row}_{col}"  # naming vars like cell_0_0, cell_0_1 etc
                
                if grid[row, col] != 0:
                    # Pre-filled cell: domain is just that value
                    problem.addVariable(var_name, [int(grid[row, col])])
                else:
                    # Empty cell: could be anything 1-9
                    problem.addVariable(var_name, list(range(1, 10)))
        
        self.stats['variables'] = size * size  # 81 vars always
        
        # Step 2: Add AllDifferent constraints
        constraint_count = 0
        
        # Row constraints: no dupes in rows
        for row in range(size):
            row_vars = [f"cell_{row}_{col}" for col in range(size)]
            problem.addConstraint(AllDifferentConstraint(), row_vars)  # alldiff = no repeats
            constraint_count += 1
        
        # Column constraints: no dupes in cols
        for col in range(size):
            col_vars = [f"cell_{row}_{col}" for row in range(size)]
            problem.addConstraint(AllDifferentConstraint(), col_vars)
            constraint_count += 1
        
        # Box constraints: no dupes in 3x3 boxes
        for box_row in range(3):
            for box_col in range(3):
                box_vars = []
                for r in range(3):
                    for c in range(3):
                        row = box_row * 3 + r  # convert box coords to grid coords
                        col = box_col * 3 + c
                        box_vars.append(f"cell_{row}_{col}")
                problem.addConstraint(AllDifferentConstraint(), box_vars)
                constraint_count += 1
        
        self.stats['constraints'] = constraint_count  # 9+9+9=27 constraints
        
        return problem
    
    def _parse_solution(self, csp_solution: Dict[str, int]) -> np.ndarray:
        """
        Parse CSP solver output back to our representation.
        Converts variable assignments to 9x9 grid.
        """
        grid = np.zeros((9, 9), dtype=int)
        
        for var_name, value in csp_solution.items():
            # Parse variable name "cell_r_c" -> extract r and c
            parts = var_name.split('_')
            row = int(parts[1])
            col = int(parts[2])
            grid[row, col] = value  # put the value where it belongs
        
        return grid
    
    def solve(self, sudoku_state: SudokuState) -> Tuple[Optional[SudokuState], dict]:
        """
        Solve Sudoku using CSP reduction.
        
        Steps:
        1. Generate CSP from Sudoku (reduction)
        2. Call CSP solver
        3. Parse solution back to SudokuState
        
        Returns:
            (solution_state, statistics) or (None, statistics) if no solution
        """
        total_start = time.time()
        
        # Model the problem as CSP (the reduction part)
        model_start = time.time()
        csp_problem = self._create_csp_from_sudoku(sudoku_state)
        self.stats['modeling_time'] = time.time() - model_start
        
        # Call solver
        solve_start = time.time()
        solution = csp_problem.getSolution()  # magic 
        self.stats['solving_time'] = time.time() - solve_start
        
        self.stats['total_time'] = time.time() - total_start
        
        # Parse solution back to original repr
        if solution is None:
            return None, self.stats  # no sol
        
        solution_grid = self._parse_solution(solution)
        solution_state = SudokuState(solution_grid)  # back to original format
        
        return solution_state, self.stats
    
    def solve_all(self, sudoku_state: SudokuState) -> Tuple[List[SudokuState], dict]:
        """
        Find ALL solutions (useful for checking if puzzle is unique).
        Warning: can be slow if many solutions exist!
        """
        total_start = time.time()
        
        csp_problem = self._create_csp_from_sudoku(sudoku_state)
        
        solve_start = time.time()
        all_solutions = csp_problem.getSolutions()  # get em all
        self.stats['solving_time'] = time.time() - solve_start
        self.stats['total_time'] = time.time() - total_start
        self.stats['num_solutions'] = len(all_solutions)  # how many I got
        
        solution_states = []
        for sol in all_solutions:
            grid = self._parse_solution(sol)
            solution_states.append(SudokuState(grid))
        
        return solution_states, self.stats


def print_comparison(initial: SudokuState, solution: SudokuState):
    """Print initial and solution side by side."""
    init_lines = str(initial).split('\n')
    sol_lines = str(solution).split('\n')
    
    print("\n" + " " * 10 + "Initial" + " " * 16 + "Solution")
    print("-" * 50)
    for i, (init_line, sol_line) in enumerate(zip(init_lines, sol_lines)):
        print(f"{init_line}    {sol_line}")


if __name__ == "__main__":
    print("=" * 60)
    print("SUDOKU CSP SOLVER")
    print("Using python-constraint library")
    print("=" * 60)
    
    # Test on all puzzles
    puzzles = [
        ("EASY", EASY_PUZZLE),
        ("MEDIUM", MEDIUM_PUZZLE),
        ("HARD", HARD_PUZZLE)
    ]
    
    solver = SudokuCSPSolver()
    
    for name, puzzle_str in puzzles:
        print(f"\n{'=' * 60}")
        print(f"Testing {name} puzzle")
        print("=" * 60)
        
        # Load puzzle
        initial = load_sudoku_from_string(puzzle_str.replace('\n', ''))
        print("\nInitial puzzle:")
        print(initial)
        print(f"\nEmpty cells: {len(initial.get_empty_cells())}")
        
        # Solve using CSP
        print("\nSolving with CSP...")
        solution, stats = solver.solve(initial)
        
        if solution:
            print("\nSolution found!")
            print(solution)
            
            # Verify solution
            print(f"\nSolution valid: {solution.is_goal()}")
            
            print(f"\nStatistics:")
            print(f"  Variables: {stats['variables']}")
            print(f"  Constraints: {stats['constraints']}")
            print(f"  Modeling time: {stats['modeling_time']*1000:.2f} ms")
            print(f"  Solving time: {stats['solving_time']*1000:.2f} ms")
            print(f"  Total time: {stats['total_time']*1000:.2f} ms")
        else:
            print("\nNo solution found!")
    
    # Demo: show CSP formulation
    print("\n" + "=" * 60)
    print("CSP FORMULATION DETAILS")
    print("=" * 60)
    print("""
    Variables: cell_r_c for each cell (r=row, c=col)
               81 variables total
    
    Domains: 
        - Pre-filled cells: {value}
        - Empty cells: {1, 2, 3, 4, 5, 6, 7, 8, 9}
    
    Constraints (27 total):
        - 9 row constraints: AllDiff(cells in row)
        - 9 column constraints: AllDiff(cells in column)  
        - 9 box constraints: AllDiff(cells in 3x3 box)
    """)

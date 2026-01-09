"""
Benchmark script for comparing A* and CSP solvers on Sudoku puzzles.
Runs both algorithms on puzzles of increasing difficulty and collects metrics.
"""

import time
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import random

from sudoku import SudokuState, load_sudoku_from_string, EASY_PUZZLE, MEDIUM_PUZZLE, HARD_PUZZLE
from a_star import AStar, SudokuProblem, empty_cells_heuristic, advanced_heuristic
from sudoku_csp import SudokuCSPSolver


# More puzzles for benchmarking - ordered by difficulty (more empty cells = harder usually)
PUZZLES = {
    "easy_1": "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
    "easy_2": "003020600900305001001806400008102900700000008006708200002609500800203009005010300",
    "medium_1": "200080300060070084030500209000105408000000000402706000301007040720040060004010003",
    "medium_2": "000000907000420180000705026100904000050000040000507009920108000034059000507000000",
    "hard_1": "800000000003600000070090200050007000000045700000100030001000068008500010090000400",
    "hard_2": "000006000059000008200008000045000000003000000006003054000325006000000000000000000",
    "expert_1": "100007090030020008009600500005300900010080002600004000300000010040000007007000300",
    "expert_2": "000000000000003085001020000000507000004000100090000000500000073002010000000040009",
}


@dataclass
class BenchmarkResult:
    """Stores results from a single benchmark run."""
    puzzle_name: str
    algorithm: str
    heuristic: str  # only for A*
    empty_cells: int  # difficulty indicator
    solved: bool
    time_elapsed: float
    # A* specific
    nodes_expanded: int = 0
    nodes_generated: int = 0
    max_frontier_size: int = 0
    max_memory_nodes: int = 0  # frontier + explored
    solution_length: int = 0
    avg_branching_factor: float = 0.0
    max_branching_factor: int = 0
    min_branching_factor: int = 0
    # CSP specific
    variables: int = 0
    constraints: int = 0
    modeling_time: float = 0.0
    solving_time: float = 0.0


class BenchmarkRunner:
    """Runs benchmarks and collects metrics."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def run_astar(self, puzzle: SudokuState, puzzle_name: str, 
                  heuristic_fn, heuristic_name: str) -> BenchmarkResult:
        """Run A* on a puzzle and collect metrics."""
        problem = SudokuProblem(puzzle)
        solver = AStarWithMetrics(heuristic_fn=heuristic_fn)  # extended version w/ more stats
        
        solution, stats = solver.search(problem)
        
        result = BenchmarkResult(
            puzzle_name=puzzle_name,
            algorithm="A*",
            heuristic=heuristic_name,
            empty_cells=len(puzzle.get_empty_cells()),
            solved=solution is not None,
            time_elapsed=stats['time_elapsed'],
            nodes_expanded=stats['nodes_expanded'],
            nodes_generated=stats['nodes_generated'],
            max_frontier_size=stats['max_frontier_size'],
            max_memory_nodes=stats.get('max_memory_nodes', 0),
            solution_length=stats.get('solution_length', 0) or 0,
            avg_branching_factor=stats.get('avg_branching_factor', 0.0),
            max_branching_factor=stats.get('max_branching_factor', 0),
            min_branching_factor=stats.get('min_branching_factor', 0)
        )
        return result
    
    def run_csp(self, puzzle: SudokuState, puzzle_name: str) -> BenchmarkResult:
        """Run CSP solver on a puzzle and collect metrics."""
        solver = SudokuCSPSolver()
        
        solution, stats = solver.solve(puzzle)
        
        result = BenchmarkResult(
            puzzle_name=puzzle_name,
            algorithm="CSP",
            heuristic="N/A",
            empty_cells=len(puzzle.get_empty_cells()),
            solved=solution is not None,
            time_elapsed=stats['total_time'],
            variables=stats['variables'],
            constraints=stats['constraints'],
            modeling_time=stats['modeling_time'],
            solving_time=stats['solving_time']
        )
        return result
    
    def run_all_benchmarks(self, puzzles: Dict[str, str], 
                           num_runs: int = 3) -> List[BenchmarkResult]:
        """Run all benchmarks multiple times and average results."""
        all_results = []
        
        # Sort puzzles by difficulty (empty cells)
        sorted_puzzles = sorted(puzzles.items(), 
                               key=lambda x: len(load_sudoku_from_string(x[1]).get_empty_cells()))
        
        for puzzle_name, puzzle_str in sorted_puzzles:
            puzzle = load_sudoku_from_string(puzzle_str)
            empty_count = len(puzzle.get_empty_cells())
            
            print(f"\n{'='*60}")
            print(f"Benchmarking: {puzzle_name} ({empty_count} empty cells)")
            print("="*60)
            
            # Run A* with simple heuristic
            print(f"\n  Running A* (empty cells heuristic)...")
            astar_simple_results = []
            for run in range(num_runs):
                result = self.run_astar(puzzle, puzzle_name, 
                                       empty_cells_heuristic, "empty_cells")
                astar_simple_results.append(result)
                print(f"    Run {run+1}: {result.time_elapsed*1000:.2f}ms, "
                      f"expanded={result.nodes_expanded}")
            all_results.append(self._average_results(astar_simple_results))
            
            # Run A* with advanced heuristic
            print(f"\n  Running A* (advanced heuristic)...")
            astar_adv_results = []
            for run in range(num_runs):
                result = self.run_astar(puzzle, puzzle_name,
                                       advanced_heuristic, "advanced")
                astar_adv_results.append(result)
                print(f"    Run {run+1}: {result.time_elapsed*1000:.2f}ms, "
                      f"expanded={result.nodes_expanded}")
            all_results.append(self._average_results(astar_adv_results))
            
            # Run CSP
            print(f"\n  Running CSP solver...")
            csp_results = []
            for run in range(num_runs):
                result = self.run_csp(puzzle, puzzle_name)
                csp_results.append(result)
                print(f"    Run {run+1}: {result.time_elapsed*1000:.2f}ms")
            all_results.append(self._average_results(csp_results))
        
        self.results = all_results
        return all_results
    
    def _average_results(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        """Average multiple benchmark results."""
        if not results:
            return None
        
        # Use first result as template
        avg = BenchmarkResult(
            puzzle_name=results[0].puzzle_name,
            algorithm=results[0].algorithm,
            heuristic=results[0].heuristic,
            empty_cells=results[0].empty_cells,
            solved=all(r.solved for r in results),
            time_elapsed=np.mean([r.time_elapsed for r in results]),
            nodes_expanded=int(np.mean([r.nodes_expanded for r in results])),
            nodes_generated=int(np.mean([r.nodes_generated for r in results])),
            max_frontier_size=int(np.mean([r.max_frontier_size for r in results])),
            max_memory_nodes=int(np.mean([r.max_memory_nodes for r in results])),
            solution_length=int(np.mean([r.solution_length for r in results])),
            avg_branching_factor=np.mean([r.avg_branching_factor for r in results]),
            max_branching_factor=int(np.max([r.max_branching_factor for r in results])),
            min_branching_factor=int(np.min([r.min_branching_factor for r in results])),
            variables=results[0].variables,
            constraints=results[0].constraints,
            modeling_time=np.mean([r.modeling_time for r in results]),
            solving_time=np.mean([r.solving_time for r in results])
        )
        return avg
    
    def print_summary(self):
        """Print a nice summary table of results."""
        print("\n" + "="*100)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*100)
        
        # Group by puzzle
        puzzles = sorted(set(r.puzzle_name for r in self.results),
                        key=lambda x: next(r.empty_cells for r in self.results if r.puzzle_name == x))
        
        for puzzle_name in puzzles:
            puzzle_results = [r for r in self.results if r.puzzle_name == puzzle_name]
            empty = puzzle_results[0].empty_cells
            
            print(f"\n{puzzle_name} ({empty} empty cells)")
            print("-"*90)
            print(f"{'Algorithm':<20} {'Heuristic':<15} {'Time(ms)':<12} {'Expanded':<12} "
                  f"{'Generated':<12} {'MaxMem':<10} {'AvgBF':<8}")
            print("-"*90)
            
            for r in puzzle_results:
                if r.algorithm == "A*":
                    print(f"{r.algorithm:<20} {r.heuristic:<15} {r.time_elapsed*1000:<12.2f} "
                          f"{r.nodes_expanded:<12} {r.nodes_generated:<12} "
                          f"{r.max_memory_nodes:<10} {r.avg_branching_factor:<8.2f}")
                else:
                    print(f"{r.algorithm:<20} {r.heuristic:<15} {r.time_elapsed*1000:<12.2f} "
                          f"{'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<8}")
        
        # Print CSP-specific metrics
        print("\n" + "="*100)
        print("CSP SOLVER SPECIFIC METRICS")
        print("="*100)
        print(f"{'Puzzle':<15} {'Empty':<8} {'Variables':<12} {'Constraints':<12} "
              f"{'Model(ms)':<12} {'Solve(ms)':<12} {'Total(ms)':<12}")
        print("-"*80)
        
        for r in self.results:
            if r.algorithm == "CSP":
                print(f"{r.puzzle_name:<15} {r.empty_cells:<8} {r.variables:<12} "
                      f"{r.constraints:<12} {r.modeling_time*1000:<12.2f} "
                      f"{r.solving_time*1000:<12.2f} {r.time_elapsed*1000:<12.2f}")
        
        # Print A* branching factor details
        print("\n" + "="*100)
        print("A* BRANCHING FACTOR DETAILS")
        print("="*100)
        print(f"{'Puzzle':<15} {'Heuristic':<15} {'MinBF':<10} {'AvgBF':<10} {'MaxBF':<10}")
        print("-"*60)
        
        for r in self.results:
            if r.algorithm == "A*":
                print(f"{r.puzzle_name:<15} {r.heuristic:<15} {r.min_branching_factor:<10} "
                      f"{r.avg_branching_factor:<10.2f} {r.max_branching_factor:<10}")
    
    def export_to_csv(self, filename: str = "benchmark_results.csv"):
        """Export results to CSV for further analysis."""
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                'puzzle_name', 'algorithm', 'heuristic', 'empty_cells', 'solved',
                'time_ms', 'nodes_expanded', 'nodes_generated', 'max_frontier',
                'max_memory', 'solution_length', 'avg_bf', 'max_bf', 'min_bf',
                'variables', 'constraints', 'model_time_ms', 'solve_time_ms'
            ])
            
            for r in self.results:
                writer.writerow([
                    r.puzzle_name, r.algorithm, r.heuristic, r.empty_cells, r.solved,
                    r.time_elapsed * 1000, r.nodes_expanded, r.nodes_generated,
                    r.max_frontier_size, r.max_memory_nodes, r.solution_length,
                    r.avg_branching_factor, r.max_branching_factor, r.min_branching_factor,
                    r.variables, r.constraints, r.modeling_time * 1000, r.solving_time * 1000
                ])
        
        print(f"\nResults exported to {filename}")


class AStarWithMetrics(AStar):
    """Extended A* that tracks additional metrics for benchmarking."""
    
    def __init__(self, heuristic_fn=None):
        super().__init__(heuristic_fn)
        self.branching_factors = []  # track bf for each expansion
        self.max_explored_size = 0
    
    def search(self, problem):
        """A* search with extra metric tracking."""
        import heapq
        from a_star import Node
        
        # Reset stats
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.max_frontier_size = 0
        self.max_explored_size = 0
        self.branching_factors = []
        start_time = time.time()
        
        # Initialize
        initial_state = problem.get_initial_state()
        h = self.get_heuristic(initial_state)
        root_node = Node(f_cost=h, state=initial_state, g_cost=0, h_cost=h)
        
        frontier = [root_node]
        heapq.heapify(frontier)
        explored = set()
        frontier_dict = {initial_state: root_node}
        
        self.nodes_generated = 1
        
        while frontier:
            # Track memory usage
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            self.max_explored_size = max(self.max_explored_size, len(explored))
            
            current_node = heapq.heappop(frontier)
            del frontier_dict[current_node.state]
            
            if problem.is_goal(current_node.state):
                end_time = time.time()
                solution = self._reconstruct_path(current_node)
                
                # Calculate branching factor stats
                avg_bf = np.mean(self.branching_factors) if self.branching_factors else 0
                max_bf = max(self.branching_factors) if self.branching_factors else 0
                min_bf = min(self.branching_factors) if self.branching_factors else 0
                
                stats = {
                    'nodes_expanded': self.nodes_expanded,
                    'nodes_generated': self.nodes_generated,
                    'max_frontier_size': self.max_frontier_size,
                    'max_memory_nodes': self.max_frontier_size + self.max_explored_size,
                    'solution_length': len(solution) - 1,
                    'solution_cost': current_node.g_cost,
                    'time_elapsed': end_time - start_time,
                    'avg_branching_factor': avg_bf,
                    'max_branching_factor': max_bf,
                    'min_branching_factor': min_bf
                }
                return solution, stats
            
            explored.add(current_node.state)
            self.nodes_expanded += 1
            
            # Get successors and track branching factor
            successors = problem.get_successors(current_node.state)
            if successors:  # only track if there are successors
                self.branching_factors.append(len(successors))
            
            for successor_state, action, step_cost in successors:
                new_g = current_node.g_cost + step_cost
                new_h = self.get_heuristic(successor_state)
                new_f = new_g + new_h
                
                if successor_state in explored:
                    continue
                
                existing_node = frontier_dict.get(successor_state)
                
                if existing_node is None:
                    new_node = Node(
                        f_cost=new_f,
                        state=successor_state,
                        g_cost=new_g,
                        h_cost=new_h,
                        parent=current_node,
                        action=action
                    )
                    heapq.heappush(frontier, new_node)
                    frontier_dict[successor_state] = new_node
                    self.nodes_generated += 1
                    
                elif new_g < existing_node.g_cost:
                    frontier.remove(existing_node)
                    heapq.heapify(frontier)
                    
                    better_node = Node(
                        f_cost=new_f,
                        state=successor_state,
                        g_cost=new_g,
                        h_cost=new_h,
                        parent=current_node,
                        action=action
                    )
                    heapq.heappush(frontier, better_node)
                    frontier_dict[successor_state] = better_node
        
        # No solution
        end_time = time.time()
        avg_bf = np.mean(self.branching_factors) if self.branching_factors else 0
        stats = {
            'nodes_expanded': self.nodes_expanded,
            'nodes_generated': self.nodes_generated,
            'max_frontier_size': self.max_frontier_size,
            'max_memory_nodes': self.max_frontier_size + self.max_explored_size,
            'solution_length': None,
            'solution_cost': None,
            'time_elapsed': end_time - start_time,
            'avg_branching_factor': avg_bf,
            'max_branching_factor': max(self.branching_factors) if self.branching_factors else 0,
            'min_branching_factor': min(self.branching_factors) if self.branching_factors else 0
        }
        return None, stats


if __name__ == "__main__":
    print("="*60)
    print("SUDOKU SOLVER BENCHMARK")
    print("Comparing A* (2 heuristics) vs CSP")
    print("="*60)
    
    runner = BenchmarkRunner()
    
    # Run benchmarks
    results = runner.run_all_benchmarks(PUZZLES, num_runs=3)
    
    # Print summary
    runner.print_summary()
    
    # Export to CSV
    runner.export_to_csv("benchmark_results.csv")
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)

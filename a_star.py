import heapq
from typing import List, Tuple, Optional, Callable, Any, Set
from dataclasses import dataclass, field
import time

@dataclass(order=True)
class Node:
    """Node for A* search tree."""
    f_cost: float  # g + h (for priority queue sorting)
    state: Any = field(compare=False)  # actual state (not used 4 comparison)
    g_cost: float = field(compare=False)  # cost from start
    h_cost: float = field(compare=False)  # heuristic estimate to goal
    parent: Optional['Node'] = field(default=None, compare=False)  # parent node for path reconstruction
    action: Any = field(default=None, compare=False)  # action that led to this state
    
    def __hash__(self):
        return hash(self.state)  # hash based on state only


class AStar:
    """A* search algorithm with duplicate elimination and no reopening."""
    
    def __init__(self, heuristic_fn: Optional[Callable] = None):
        """
        Initialize A* solver.
        Args:
            heuristic_fn: function that takes a state and returns h value.
                         If None, uses state.heuristic() method
        """
        self.heuristic_fn = heuristic_fn
        self.nodes_expanded = 0  # stats tracking
        self.nodes_generated = 0
        self.max_frontier_size = 0
        
    def get_heuristic(self, state) -> float:
        """Get heuristic value for a state."""
        if self.heuristic_fn is not None:
            return self.heuristic_fn(state)  # use custom heuristic if provided
        else:
            return state.heuristic()  # otherwise use state's own method
    
    def search(self, problem) -> Tuple[Optional[List], dict]:
        """
        Perform A* search on the problem.
        Args:
            problem: must have methods:
                - get_initial_state() -> state
                - is_goal(state) -> bool
                - get_successors(state) -> list of (new_state, action, cost)
        Returns:
            (solution_path, statistics) or (None, statistics) if no solution
        """
        # Reset stats
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.max_frontier_size = 0
        start_time = time.time()
        
        # Initialize with start state
        initial_state = problem.get_initial_state()
        h = self.get_heuristic(initial_state)
        root_node = Node(f_cost=h, state=initial_state, g_cost=0, h_cost=h)
        
        # frontier = priority queue ordered by f = g + h
        frontier = [root_node]  # heapq will maintain min-heap property
        heapq.heapify(frontier)
        
        # explored = set of already expanded states
        explored: Set = set()
        
        # frontier_dict for quick lookup of states in frontier
        frontier_dict = {initial_state: root_node}  # state -> node mapping
        
        self.nodes_generated = 1
        
        # Main A* loop
        while frontier:
            # Track max frontier size
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            
            # Pop node w/ lowest f-cost
            current_node = heapq.heappop(frontier)
            del frontier_dict[current_node.state]  # remove from dict too
            
            # Goal test
            if problem.is_goal(current_node.state):
                end_time = time.time()
                solution = self._reconstruct_path(current_node)
                stats = {
                    'nodes_expanded': self.nodes_expanded,
                    'nodes_generated': self.nodes_generated,
                    'max_frontier_size': self.max_frontier_size,
                    'solution_length': len(solution) - 1,  # -1 bc we count actions not states
                    'solution_cost': current_node.g_cost,
                    'time_elapsed': end_time - start_time
                }
                return solution, stats
            
            # Add to explored
            explored.add(current_node.state)
            self.nodes_expanded += 1
            
            # Expand node - get successors
            for successor_state, action, step_cost in problem.get_successors(current_node.state):
                # Calculate costs
                new_g = current_node.g_cost + step_cost
                new_h = self.get_heuristic(successor_state)
                new_f = new_g + new_h
                
                # Check if state already explored or in frontier
                if successor_state in explored:
                    continue  # skip if already explored (no reopening!)
                
                existing_node = frontier_dict.get(successor_state)
                
                if existing_node is None:
                    # New state - add to frontier
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
                    # Found better path to existing frontier node
                    # Remove old node and add new one w/ better cost
                    frontier.remove(existing_node)  # remove old version
                    heapq.heapify(frontier)  # re-heapify after removal
                    
                    better_node = Node(
                        f_cost=new_f,
                        state=successor_state,
                        g_cost=new_g,
                        h_cost=new_h,
                        parent=current_node,
                        action=action
                    )
                    heapq.heappush(frontier, better_node)
                    frontier_dict[successor_state] = better_node  # update dict
        
        # No solution found
        end_time = time.time()
        stats = {
            'nodes_expanded': self.nodes_expanded,
            'nodes_generated': self.nodes_generated,
            'max_frontier_size': self.max_frontier_size,
            'solution_length': None,
            'solution_cost': None,
            'time_elapsed': end_time - start_time
        }
        return None, stats
    
    def _reconstruct_path(self, goal_node: Node) -> List:
        """Reconstruct solution path from goal to start."""
        path = []
        current = goal_node
        
        # Trace back through parents
        while current is not None:
            path.append((current.state, current.action))
            current = current.parent
        
        path.reverse()  # reverse to get start -> goal
        return path


class SudokuProblem:
    """Wrapper to make SudokuState compatible w/ A* interface."""
    
    def __init__(self, initial_state):
        self.initial_state = initial_state
    
    def get_initial_state(self):
        return self.initial_state
    
    def is_goal(self, state) -> bool:
        return state.is_goal()
    
    def get_successors(self, state) -> List[Tuple[Any, Any, float]]:
        """
        Returns list of (successor_state, action, cost).
        For Sudoku, all moves have cost 1.
        """
        successors = []
        for successor_state, action in state.get_successors():
            successors.append((successor_state, action, 1.0))  # uniform cost = 1
        return successors


# Custom heuristic examples
def empty_cells_heuristic(state) -> float:
    """Simple heuristic: count empty cells."""
    return state.heuristic()  # uses built-in method


def advanced_heuristic(state) -> float:
    """
    Advanced heuristic: weighted sum of:
    - empty cells
    - cells w/ few possibilities (harder to fill)
    """
    empty_count = state.heuristic()
    
    # Add penalty for cells w/ very few options (likely to cause conflicts)
    constraint_penalty = 0
    for row, col in state.get_empty_cells():
        possible = len(state.get_possible_values(row, col))
        if possible == 0:
            return float('inf')  # dead end
        elif possible == 1:
            constraint_penalty += 0  # easy to fill
        elif possible == 2:
            constraint_penalty += 0.5
        else:
            constraint_penalty += 1
    
    return empty_count + constraint_penalty * 0.1  # weight the penalty


if __name__ == "__main__":
    from sudoku import load_sudoku_from_string, EASY_PUZZLE, MEDIUM_PUZZLE, HARD_PUZZLE
    
    # Test A* w/ easy puzzle
    print("=" * 50)
    print("Testing A* on EASY puzzle")
    print("=" * 50)
    
    initial = load_sudoku_from_string(EASY_PUZZLE.replace('\n', ''))
    print("\nInitial state:")
    print(initial)
    
    problem = SudokuProblem(initial)
    solver = AStar(heuristic_fn=empty_cells_heuristic)
    
    print("\nSearching...")
    solution, stats = solver.search(problem)
    
    if solution:
        print("\n✓ Solution found!")
        print(f"\nFinal state:")
        print(solution[-1][0])  # last state in path
        
        print(f"\nStatistics:")
        print(f"  Nodes expanded: {stats['nodes_expanded']}")
        print(f"  Nodes generated: {stats['nodes_generated']}")
        print(f"  Max frontier size: {stats['max_frontier_size']}")
        print(f"  Solution length: {stats['solution_length']} moves")
        print(f"  Solution cost: {stats['solution_cost']}")
        print(f"  Time: {stats['time_elapsed']:.3f} seconds")
    else:
        print("\n✗ No solution found")
        print(f"Nodes expanded: {stats['nodes_expanded']}")

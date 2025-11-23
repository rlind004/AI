import heapq
import time

# --- Constants for Grid Representation ---
EMPTY_SPACE = 0
ASTEROID = 1
STATION = 2

class SearchNode:
    """
    Represents a node in the search tree.
    - state: A tuple representing the agent's state (x, y, fuel).
    - parent: The node that generated this node.
    - action: The action that was taken to get to this state.
    - cost: The cumulative cost (g(n)) from the start node to this node.
    - f_cost: The estimated total cost (f(n) = g(n) + h(n)) for A* search.
    """
    def __init__(self, state, parent=None, action=None, cost=0, f_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.f_cost = f_cost

    def __lt__(self, other):
        """
        Comparison method for the priority queue. The priority queue will pop the
        node with the lowest f_cost. This is essential for both UCS and A*.
        """
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        """
        Equality comparison for states, used to check if a state has been visited.
        """
        return self.state == other.state

    def __hash__(self):
        """
        Hash method for states, allowing nodes to be stored in a set (the 'explored' set).
        """
        return hash(self.state)

class AsteroidFieldProblem:
    """
    Defines the "Escape from the Asteroid Field" problem.
    - grid: A 2D list representing the field.
    - start_pos: The (x, y) starting coordinates of the spaceship.
    - max_fuel: The maximum fuel capacity of the spaceship.
    """
    def __init__(self, grid, start_pos, max_fuel):
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0])
        self.max_fuel = max_fuel
        # The initial state includes the starting position and maximum fuel.
        self.initial_state = (start_pos[0], start_pos[1], max_fuel)

    def get_successors(self, state):
        """
        Generates all valid successor states from the current state.
        A successor is a valid move to an adjacent cell.
        Returns a list of (next_state, action, cost) tuples.
        """
        x, y, fuel = state
        successors = []
        
        # Defines possible moves: (dx, dy, move_name, fuel_cost)
        # Diagonal moves cost more fuel.
        moves = [
            (0, 1, 'Down', 1), (0, -1, 'Up', 1),
            (1, 0, 'Right', 1), (-1, 0, 'Left', 1),
            (1, 1, 'Down-Right', 2), (-1, 1, 'Down-Left', 2),
            (1, -1, 'Up-Right', 2), (-1, -1, 'Up-Left', 2)
        ]

        for dx, dy, action, cost in moves:
            nx, ny = x + dx, y + dy

            # Check if the move is within grid boundaries
            if 0 <= nx < self.width and 0 <= ny < self.height:
                # Check if the cell is not an asteroid and we have enough fuel
                if self.grid[ny][nx] != ASTEROID and fuel >= cost:
                    next_fuel = self.max_fuel if self.grid[ny][nx] == STATION else fuel - cost
                    next_state = (nx, ny, next_fuel)
                    successors.append((next_state, action, cost))
        
        return successors

    def is_goal(self, state):
        """
        The goal is to reach the other side of the grid (the last row).
        """
        _, y, _ = state
        return y == self.height - 1

    def heuristic(self, state):
        """
        Heuristic function (h(n)) for A* search.
        Estimates the cost to get from the current state to the goal.
        
        Heuristic Used: Manhattan Distance
        The estimated cost is the number of vertical steps needed to reach the goal row.
        This is an admissible heuristic because it never overestimates the actual cost,
        as each vertical move costs at least 1 unit of fuel.
        """
        _, y, _ = state
        return (self.height - 1 - y)

def general_search(problem, use_astar=False):
    """
    A general search algorithm that can perform Uniform-Cost Search or A* Search.
    
    - problem: An instance of AsteroidFieldProblem.
    - use_astar: If True, uses the A* heuristic. If False, performs UCS.
    """
    start_time = time.time()

    # Create the root node of the search tree
    initial_node = SearchNode(state=problem.initial_state, cost=0)
    if use_astar:
        initial_node.f_cost = problem.heuristic(problem.initial_state)

    # The 'frontier' is a priority queue, ordered by f_cost.
    # For UCS, f_cost = g(n). For A*, f_cost = g(n) + h(n).
    frontier = [initial_node]
    
    # The 'explored' set stores states that have already been visited to avoid cycles
    # and redundant computations. We store states, not nodes.
    explored = set()

    nodes_expanded = 0

    while frontier:
        # Pop the node with the lowest f_cost from the priority queue
        current_node = heapq.heappop(frontier)
        nodes_expanded += 1

        # If the goal is reached, reconstruct and return the solution path
        if problem.is_goal(current_node.state):
            path = []
            while current_node:
                # Unpack state for better readability in the path
                pos = (current_node.state[0], current_node.state[1])
                fuel = current_node.state[2]
                path.append({'action': current_node.action, 'pos': pos, 'fuel': fuel})
                current_node = current_node.parent
            
            end_time = time.time()
            return {
                "path": list(reversed(path)),
                "cost": path[0]['cost'],  # Cost is stored in the final node
                "time": end_time - start_time,
                "nodes_expanded": nodes_expanded
            }

        # Add the current state to the explored set
        explored.add(current_node.state)

        # Expand the current node by generating its successors
        for next_state, action, cost in problem.get_successors(current_node.state):
            if next_state not in explored:
                # Calculate the cumulative cost (g(n)) for the successor
                new_cost = current_node.cost + cost
                
                # Calculate the f_cost (g(n) + h(n))
                f_cost = new_cost
                if use_astar:
                    f_cost += problem.heuristic(next_state)
                
                # Create the successor node
                successor_node = SearchNode(
                    state=next_state,
                    parent=current_node,
                    action=action,
                    cost=new_cost,
                    f_cost=f_cost
                )
                
                # Add the successor to the frontier
                heapq.heappush(frontier, successor_node)

    # If the frontier becomes empty and no solution was found
    end_time = time.time()
    return {
        "path": None,
        "cost": -1,
        "time": end_time - start_time,
        "nodes_expanded": nodes_expanded
    }

# --- Main Execution ---
if __name__ == "__main__":
    # Define a custom grid: 15x15
    # S = Start, E = Empty, A = Asteroid, F = Fuel Station
    # The actual grid will use the constants defined at the top.
    grid_layout = [
        ['S', 'E', 'E', 'E', 'A', 'A', 'A', 'E', 'E', 'E', 'E', 'E', 'A', 'E', 'E'],
        ['E', 'A', 'A', 'E', 'E', 'E', 'A', 'E', 'A', 'E', 'A', 'E', 'A', 'E', 'E'],
        ['E', 'A', 'A', 'E', 'A', 'E', 'E', 'E', 'A', 'E', 'A', 'E', 'A', 'A', 'E'],
        ['E', 'E', 'E', 'E', 'A', 'E', 'A', 'E', 'E', 'E', 'A', 'E', 'E', 'E', 'E'],
        ['A', 'A', 'E', 'A', 'A', 'A', 'A', 'E', 'A', 'A', 'A', 'A', 'E', 'A', 'E'],
        ['E', 'E', 'E', 'A', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'F', 'E', 'A', 'E'],
        ['E', 'A', 'A', 'A', 'E', 'A', 'A', 'A', 'A', 'A', 'E', 'E', 'E', 'A', 'A'],
        ['E', 'E', 'E', 'E', 'E', 'A', 'E', 'E', 'E', 'E', 'E', 'E', 'A', 'E', 'E'],
        ['E', 'A', 'A', 'E', 'A', 'A', 'A', 'E', 'A', 'E', 'A', 'A', 'A', 'E', 'E'],
        ['F', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'A', 'E', 'E', 'E', 'A', 'E', 'F'],
        ['E', 'A', 'A', 'E', 'A', 'A', 'E', 'A', 'A', 'A', 'A', 'E', 'E', 'E', 'E'],
        ['E', 'E', 'A', 'E', 'E', 'E', 'E', 'E', 'E', 'A', 'E', 'E', 'E', 'A', 'E'],
        ['E', 'A', 'A', 'E', 'A', 'E', 'A', 'E', 'A', 'A', 'E', 'A', 'E', 'A', 'E'],
        ['E', 'E', 'E', 'E', 'A', 'E', 'A', 'E', 'E', 'E', 'E', 'A', 'E', 'E', 'E'],
        ['G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G'] # Goal Row
    ]

    # Convert layout to numerical grid and find start position
    start_pos = (0, 0)
    grid = []
    for r, row in enumerate(grid_layout):
        numeric_row = []
        for c, char in enumerate(row):
            if char == 'S':
                start_pos = (c, r)
                numeric_row.append(EMPTY_SPACE)
            elif char == 'A':
                numeric_row.append(ASTEROID)
            elif char == 'F':
                numeric_row.append(STATION)
            else:
                numeric_row.append(EMPTY_SPACE)
        grid.append(numeric_row)
    
    # Initialize the problem with the grid, start position, and max fuel
    problem = AsteroidFieldProblem(grid, start_pos=start_pos, max_fuel=20)
    
    print("--- Running Uniform-Cost Search (UCS) ---")
    ucs_result = general_search(problem, use_astar=False)
    if ucs_result["path"]:
        print(f"Solution Found!")
        print(f"  - Path Cost: {ucs_result['cost']}")
        print(f"  - Nodes Expanded: {ucs_result['nodes_expanded']}")
        print(f"  - Execution Time: {ucs_result['time']:.5f} seconds")
    else:
        print("No solution found.")

    print("\n" + "="*40 + "\n")

    print("--- Running A* Search ---")
    astar_result = general_search(problem, use_astar=True)
    if astar_result["path"]:
        print(f"Solution Found!")
        print(f"  - Path Cost: {astar_result['cost']}")
        print(f"  - Nodes Expanded: {astar_result['nodes_expanded']}")
        print(f"  - Execution Time: {astar_result['time']:.5f} seconds")
    else:
        print("No solution found.")

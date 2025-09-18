
import math
import time
import heapq
import copy
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# -------------------------
# Utility functions
# -------------------------
def reconstruct_path(came_from, start, goal):
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came_from.get(cur)
    path.reverse()
    if len(path) > 0 and path[0] == start:
        return path
    return None

# -------------------------
# Grid and dynamic obstacles
# -------------------------
class DynamicObstacle:
    """
    Moves along a given list of positions cyclically (one step per timestep).
    positions: list of (r,c)
    start_time: offset at which obstacle is at positions[0]
    """
    def __init__(self, positions, start_time=0):
        self.positions = positions
        self.start_time = start_time

    def position_at(self, t):
        if len(self.positions) == 0:
            return None
        idx = (t - self.start_time) % len(self.positions)
        return self.positions[idx]

class Grid:
    """
    Grid representation:
      - static obstacles: cost = -1
      - terrain cost: integer >= 1
      - dynamic obstacles: list of DynamicObstacle objects
    """
    def __init__(self, matrix, dynamic_obstacles=None, allow_diagonals=False):
        self.matrix = np.array(matrix)
        self.rows, self.cols = self.matrix.shape
        self.dynamic_obstacles = dynamic_obstacles or []
        self.diagonals = allow_diagonals

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_static_obstacle(self, r, c):
        return self.matrix[r, c] == -1

    def is_blocked(self, r, c, t=0):
        """
        True if blocked at time t (either static obstacle or dynamic obstacle occupies it)
        """
        if not self.in_bounds(r, c):
            return True
        if self.is_static_obstacle(r, c):
            return True
        # check dynamic obstacles
        for dob in self.dynamic_obstacles:
            pos = dob.position_at(t)
            if pos == (r, c):
                return True
        return False

    def cost(self, r, c):
        """Return terrain movement cost (>=1). For obstacles returns inf."""
        if not self.in_bounds(r, c):
            return math.inf
        if self.is_static_obstacle(r, c):
            return math.inf
        val = int(self.matrix[r, c])
        return max(1, val)

    def neighbors(self, r, c):
        steps = [(1,0), (-1,0), (0,1), (0,-1)]
        if self.diagonals:
            steps += [(1,1),(1,-1),(-1,1),(-1,-1)]
        for dr, dc in steps:
            nr, nc = r+dr, c+dc
            if self.in_bounds(nr, nc) and not self.is_static_obstacle(nr, nc):
                yield (nr, nc)

# -------------------------
# Search algorithms
# -------------------------
def bfs_search(grid, start, goal):
    """
    Classic BFS (uninformed, treats each move cost as uniform)
    Returns path, nodes_expanded
    """
    queue = deque([start])
    visited = {start: None}
    nodes_expanded = 0
    while queue:
        cur = queue.popleft()
        nodes_expanded += 1
        if cur == goal:
            return reconstruct_path(visited, start, goal), nodes_expanded
        for nb in grid.neighbors(*cur):
            if nb not in visited:
                visited[nb] = cur
                queue.append(nb)
    return None, nodes_expanded

def astar_search(grid, start, goal, time_offset=0):
    """
    A* search that uses terrain costs and Manhattan heuristic (admissible for 4-connected)
    time_offset is only used if you want to treat dynamic obstacles in the planning horizon.
    For this implementation we assume planning ignores future dynamic obstacles (common assumption),
    but we provide time_offset parameter to allow planning with a time horizon (not used below).
    Returns path, nodes_expanded, g_cost
    """
    def heuristic(a, b):
        # Manhattan distance (admissible for 4-connected grid)
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    open_heap = []
    heapq.heappush(open_heap, (0 + heuristic(start, goal), 0, start))  # (f, g, node)
    came_from = {start: None}
    gscore = {start: 0}
    nodes_expanded = 0
    visited = set()

    while open_heap:
        f, g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return path, nodes_expanded, gscore[goal]

        for nb in grid.neighbors(*current):
            # The search here ignores dynamic obstacles (commonly done for A* planning)
            tentative_g = gscore[current] + grid.cost(*nb)
            if nb not in gscore or tentative_g < gscore[nb]:
                gscore[nb] = tentative_g
                fscore = tentative_g + heuristic(nb, goal)
                heapq.heappush(open_heap, (fscore, tentative_g, nb))
                came_from[nb] = current

    return None, nodes_expanded, math.inf

# -------------------------
# Agent with replanning
# -------------------------
class Agent:
    """
    Agent that follows a planned path step-by-step and replans (using A*) when next step blocked.
    Simulation is discrete time steps; dynamic obstacles move 1 step per time unit.
    """
    def __init__(self, grid, start, goal, planner='astar'):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.planner = planner  # 'astar' or 'bfs'
        self.path = []
        self.path_index = 0
        self.time = 0
        self.logs = []  # store events (time, event string)
        self.nodes_expanded_total = 0
        self.replans = 0

    def plan(self):
        if self.planner == 'bfs':
            p, nodes = bfs_search(self.grid, self.start, self.goal)
            self.nodes_expanded_total += nodes
            self.logs.append((self.time, f"Initial BFS plan nodes_expanded={nodes}"))
            return p
        else:
            p, nodes, cost = astar_search(self.grid, self.start, self.goal, time_offset=self.time)
            self.nodes_expanded_total += nodes
            self.logs.append((self.time, f"Initial A* plan nodes_expanded={nodes} cost={cost}"))
            return p

    def step(self):
        """
        Move one step along path. If next step becomes blocked at current time+1 (since movement occurs),
        replan using A*. If no path, try to plan. Returns True if reached goal, False otherwise.
        """
        # If no path, attempt to plan
        if not self.path:
            self.path = self.plan()
            self.path_index = 0
            if not self.path:
                self.logs.append((self.time, "No path found on planning"))
                return False

        # If already at goal
        cur_pos = self.path[self.path_index]
        if cur_pos == self.goal:
            self.logs.append((self.time, "Already at goal"))
            return True

        # Determine next step
        if self.path_index + 1 >= len(self.path):
            # path ended but not at goal (shouldn't happen)
            self.logs.append((self.time, "Path ended unexpectedly"))
            return False

        next_pos = self.path[self.path_index + 1]

        # check if next_pos will be blocked at time+1 (when agent arrives)
        if self.grid.is_blocked(next_pos[0], next_pos[1], self.time + 1):
            # need to replan from current position at time+1
            self.replans += 1
            self.logs.append((self.time, f"Next cell {next_pos} blocked at t={self.time+1}. Replanning..."))
            # update start to current position
            current_position = self.path[self.path_index]
            # For replanning it's reasonable to treat dynamic obstacles' current/future positions at time+1
            # We'll temporarily advance simulation time by 0 and call planner. For simplicity, planner ignores dynamic obstacles;
            # but we still check for immediate blocking.
            p, nodes, cost = astar_search(self.grid, current_position, self.goal, time_offset=self.time+1)
            self.nodes_expanded_total += nodes
            if p:
                self.path = p
                self.path_index = 0
                self.logs.append((self.time, f"Replan success nodes_expanded={nodes} cost={cost}"))
                # do not advance time here; proceed to attempt moving in next loop iteration
                return False
            else:
                self.logs.append((self.time, "Replan failed (no path)"))
                return False

        # move to next_pos
        self.path_index += 1
        self.time += 1
        self.logs.append((self.time, f"Moved to {next_pos}"))
        # if reached goal
        if next_pos == self.goal:
            total_cost = self.total_path_cost()
            self.logs.append((self.time, f"Reached goal at t={self.time} cost={total_cost} replans={self.replans}"))
            return True
        return False

    def total_path_cost(self):
        if not self.path:
            return math.inf
        return sum(self.grid.cost(r, c) for r, c in self.path)

# -------------------------
# Visualization helpers
# -------------------------
def plot_grid(ax, grid, agent_pos=None, start=None, goal=None, path=None, title=None, time_step=None):
    cmap = colors.ListedColormap(['black', 'white', 'lightgrey', 'sandybrown', 'lightgreen'])
    # We'll map values:
    # -1 (static obstacle) -> black
    # 1 (free) -> white
    # 2 -> lightgrey (higher terrain cost), 3 -> sandybrown, etc.
    # But since costs can vary, we'll normalize.
    mat = np.copy(grid.matrix).astype(float)
    # Replace -1 with -1, others >=1
    display = np.full(mat.shape, 1.0)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            if mat[r,c] == -1:
                display[r,c] = -1.0
            else:
                display[r,c] = mat[r,c]  # terrain cost
    # create color mapping: obstacles black, cost=1 white, higher costs as shades
    ax.clear()
    ax.set_xticks(np.arange(-.5, grid.cols, 1), minor=False)
    ax.set_yticks(np.arange(-.5, grid.rows, 1), minor=False)
    ax.grid(which='major', color='lightgrey', linestyle='-', linewidth=0.5)
    ax.set_xlim(-0.5, grid.cols-0.5)
    ax.set_ylim(grid.rows-0.5, -0.5)  # invert y for natural matrix orientation

    # draw terrain cells
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.is_static_obstacle(r,c):
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='black'))
            else:
                cost_val = int(grid.matrix[r,c])
                if cost_val <= 1:
                    face = 'white'
                elif cost_val == 2:
                    face = 'lightgrey'
                elif cost_val == 3:
                    face = 'sandybrown'
                else:
                    face = 'lightgreen'
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color=face, ec='none'))

    # draw dynamic obstacles at this time step
    if time_step is not None:
        for dob in grid.dynamic_obstacles:
            pos = dob.position_at(time_step)
            if pos:
                r, c = pos
                ax.add_patch(plt.Circle((c, r), 0.25, color='darkred'))

    # draw path
    if path:
        # convert to x,y positions (col,row)
        xs = [c for (r,c) in path]
        ys = [r for (r,c) in path]
        # but for plotting we want (x=c, y=r)
        px = [c for (r,c) in path]
        py = [r for (r,c) in path]
        ax.plot(px, py, linestyle='-', linewidth=2, marker='o', markersize=4, color='blue', alpha=0.8)

    # draw start and goal
    if start:
        ax.add_patch(plt.Rectangle((start[1]-0.5, start[0]-0.5), 1, 1, edgecolor='green', fill=False, linewidth=2))
        ax.text(start[1], start[0], 'S', ha='center', va='center', color='green', weight='bold')
    if goal:
        ax.add_patch(plt.Rectangle((goal[1]-0.5, goal[0]-0.5), 1, 1, edgecolor='gold', fill=False, linewidth=2))
        ax.text(goal[1], goal[0], 'G', ha='center', va='center', color='gold', weight='bold')

    # draw agent
    if agent_pos:
        ax.add_patch(plt.Circle((agent_pos[1], agent_pos[0]), 0.2, color='red'))

    if title:
        ax.set_title(title)
    ax.set_aspect('equal')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# -------------------------
# Example maps
# -------------------------
def small_map():
    # 7x7 grid, some obstacles
    M = np.ones((7,7), dtype=int)
    # place some static obstacles
    M[2,1:5] = -1
    M[4,2] = -1
    # add varied costs
    M[0,4] = 2
    M[1,4] = 2
    return M

def medium_map():
    # 12x12 with a wall and some higher cost terrain
    M = np.ones((12,12), dtype=int)
    M[5,1:11] = -1
    M[5,6] = 1  # hole
    M[8,3:5] = -1
    M[2:4,9] = 3
    M[10,5:8] = 2
    return M

def large_map():
    # 18x18 with some obstacles and different costs
    M = np.ones((18,18), dtype=int)
    for i in range(3,15):
        M[9,i] = -1
    M[9,9] = 1  # break
    M[4,7:10] = -1
    M[13,2:6] = -1
    # scattered higher-cost cells
    M[2,2] = 3
    M[3,3] = 3
    M[16,16] = 2
    M[1,16] = 2
    return M

def dynamic_map():
    # 12x12 base matrix
    M = np.ones((12,12), dtype=int)
    # static obstacles
    M[3,3:9] = -1
    M[7,2:6] = -1
    # terrain costs
    M[1,8] = 2
    M[2,8] = 2
    # create moving obstacle that moves horizontally across a corridor
    # moving path: (5,0) -> (5,1) -> ... -> (5,11) -> loop
    mv_positions = [(5, c) for c in range(0,12)]
    moving_vehicle = DynamicObstacle(mv_positions, start_time=0)
    # another vehicle moving vertically
    mv2_positions = [(r, 6) for r in range(11, -1, -1)]
    moving_vehicle2 = DynamicObstacle(mv2_positions, start_time=3)
    return M, [moving_vehicle, moving_vehicle2]

# -------------------------
# Run simulation & visualization
# -------------------------
def run_demo(map_choice='dynamic', planner='astar', visualize=True, pause_time=0.3):
    # choose map
    if map_choice == 'small':
        M = small_map()
        dyn = []
        start = (6, 0)
        goal = (0, 6)
    elif map_choice == 'medium':
        M = medium_map()
        dyn = []
        start = (11, 0)
        goal = (0, 11)
    elif map_choice == 'large':
        M = large_map()
        dyn = []
        start = (17, 0)
        goal = (0, 17)
    else:
        M, dyn = dynamic_map()
        start = (11, 0)
        goal = (0, 11)

    grid = Grid(M, dynamic_obstacles=dyn, allow_diagonals=False)
    agent = Agent(grid, start, goal, planner=planner)
    agent.path = agent.plan()  # initial plan
    if not agent.path:
        print("No initial path found.")
        return

    # setup plotting
    if visualize:
        fig, ax = plt.subplots(figsize=(6,6))
        plt.ion()
        plt.show()

    # simulation loop: step until goal or time cap
    max_steps = grid.rows * grid.cols * 4
    reached = False
    step_count = 0
    while step_count < max_steps:
        cur_pos = agent.path[agent.path_index] if agent.path else agent.start
        # draw
        if visualize:
            plot_grid(ax, grid, agent_pos=cur_pos, start=start, goal=goal, path=agent.path, title=f"Time {agent.time}", time_step=agent.time)
            plt.draw()
            plt.pause(pause_time)

        reached = agent.step()
        step_count += 1
        # If agent found new path (replanned) we want to visualize next iteration

        if reached:
            break

    # final drawing
    if visualize:
        final_pos = agent.path[agent.path_index] if agent.path else agent.start
        plot_grid(ax, grid, agent_pos=final_pos, start=start, goal=goal, path=agent.path, title=f"Final t={agent.time}", time_step=agent.time)
        plt.draw()
        plt.pause(1.0)
        plt.ioff()

    # print logs & statistics
    print("=== Simulation Summary ===")
    print(f"Map: {map_choice} planner: {planner}")
    print(f"Start: {start} Goal: {goal}")
    print(f"Time steps elapsed: {agent.time}")
    print(f"Total nodes expanded (accumulated across plans): {agent.nodes_expanded_total}")
    print(f"Number of replans: {agent.replans}")
    if agent.path:
        print(f"Final path cost (path length includes start): {agent.total_path_cost():.1f}")
        print(f"Path length (cells): {len(agent.path)}")
    print("\nEvent log (time, event):")
    for t, e in agent.logs:
        print(f" t={t}: {e}")

# -------------------------
# CLI-ish entrypoint
# -------------------------
if __name__ == "__main__":
    # Example runs: you can change parameters here
    print("Demo: dynamic map with A* (replanning enabled). Close the plot to proceed.")
    run_demo(map_choice='dynamic', planner='astar', visualize=True, pause_time=0.25)

    print("\nNow BFS on small map (no dynamic obstacles).")
    run_demo(map_choice='small', planner='bfs', visualize=True, pause_time=0.15)

    print("\nNow A* on medium map.")
    run_demo(map_choice='medium', planner='astar', visualize=True, pause_time=0.15)

    print("\nNow A* on large map.")
    run_demo(map_choice='large', planner='astar', visualize=True, pause_time=0.08)

    print("\nAll demos complete.")
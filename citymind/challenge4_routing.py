"""
CityMind - Challenge 4: Emergency Routing Under Changing Conditions
====================================================================
Technique : A* Search with Manhattan Distance heuristic

Why A* over plain Dijkstra:
  Dijkstra expands in ALL directions equally — O(V log V) regardless.
  A* uses a heuristic h(n) to guide search TOWARD the goal, dramatically
  reducing nodes expanded in practice while retaining the optimality guarantee.

Admissibility guarantee:
  Manhattan distance NEVER overestimates true cost on a grid (no diagonal moves).
  Because h(n) <= true cost, A* is ADMISSIBLE and always finds the shortest path.

Dynamic re-routing:
  When any road becomes blocked (flood / VIP movement / protest), the current
  path is immediately discarded and A* re-runs from the team's current position
  using the CURRENT state of the shared graph (blocked edges are marked there).
  No stale path cache — A* always reads live graph state.

Multi-civilian routing:
  The team visits civilians in sequence (nearest-unrescued ordering).
  After each rescue, A* re-plans to the next target.
"""

import heapq
import math
from city_graph import CityGraph


class Civilian:
    def __init__(self, civ_id: int, node_id: int):
        self.civ_id   = civ_id
        self.node_id  = node_id
        self.rescued  = False

    def __repr__(self):
        status = "✓ rescued" if self.rescued else "◌ waiting"
        r, c = divmod(self.node_id, 12)
        return f"Civilian {self.civ_id} @ ({r},{c}) [{status}]"


class EmergencyRouter:
    def __init__(self, graph: CityGraph):
        self.graph      = graph
        self.team_pos   = 0              # current node_id of medical team
        self.current_path: list[int] = []
        self.civilians: list[Civilian] = []
        self.log: list[str] = []

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #
    def set_team_start(self, node_id: int):
        self.team_pos = node_id
        r, c = divmod(node_id, self.graph.GRID)
        self._log(f"Medical team starting at node {node_id} ({r},{c})")

    def add_civilian(self, node_id: int):
        civ = Civilian(len(self.civilians) + 1, node_id)
        self.civilians.append(civ)
        r, c = divmod(node_id, self.graph.GRID)
        self._log(f"Civilian {civ.civ_id} registered at node {node_id} ({r},{c})")

    # ------------------------------------------------------------------ #
    #  A* Search
    # ------------------------------------------------------------------ #
    def astar(self, src: int, dst: int) -> list[int] | None:
        """
        Returns shortest path [src, ..., dst] respecting current blocked edges.
        Returns None if no path exists.
        Heuristic h(n) = Manhattan distance — admissible on grid.
        """
        G = self.graph.GRID

        def h(node_id: int) -> float:
            ra, ca = divmod(node_id, G)
            rb, cb = divmod(dst, G)
            return abs(ra - rb) + abs(ca - cb)

        g_cost = {src: 0.0}
        f_cost = {src: h(src)}
        came_from: dict[int, int] = {}
        open_heap = [(f_cost[src], src)]   # (f, node)
        closed = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == dst:
                return self._reconstruct(came_from, dst)
            closed.add(current)

            for nb, edge in self.graph.neighbors(current):
                if nb in closed:
                    continue
                # Edge cost includes base road cost + crime risk multiplier
                move_cost = edge.effective_cost * self.graph.nodes[nb].crime_multiplier
                tentative_g = g_cost[current] + move_cost

                if tentative_g < g_cost.get(nb, math.inf):
                    came_from[nb] = current
                    g_cost[nb]    = tentative_g
                    f_cost[nb]    = tentative_g + h(nb)
                    heapq.heappush(open_heap, (f_cost[nb], nb))

        return None   # No path found

    def _reconstruct(self, came_from: dict, dst: int) -> list[int]:
        path = [dst]
        while dst in came_from:
            dst = came_from[dst]
            path.append(dst)
        return list(reversed(path))

    # ------------------------------------------------------------------ #
    #  Route to next unrescued civilian
    # ------------------------------------------------------------------ #
    def plan_next(self) -> bool:
        """Plan route to the nearest unrescued civilian. Returns False if done."""
        remaining = [c for c in self.civilians if not c.rescued]
        if not remaining:
            self._log("All civilians rescued! Mission complete.")
            return False

        # Pick nearest by Manhattan distance (greedy ordering)
        target = min(remaining, key=lambda c: self._manhattan(self.team_pos, c.node_id))
        path = self.astar(self.team_pos, target.node_id)

        if path is None:
            self._log(
                f"WARNING: No path to Civilian {target.civ_id} at node {target.node_id}. "
                f"Waiting for road conditions to improve."
            )
            self.current_path = []
            return True

        self.current_path = path
        r, c = divmod(target.node_id, self.graph.GRID)
        self._log(
            f"A* path planned to Civilian {target.civ_id} @ ({r},{c}). "
            f"Path length: {len(path)} hops."
        )
        return True

    # ------------------------------------------------------------------ #
    #  Called every simulation step — move one step along path
    # ------------------------------------------------------------------ #
    def step(self) -> str:
        """
        Advance team one step. Returns status string.
        Checks for rescues after each move.
        """
        if not self.current_path or len(self.current_path) < 2:
            self.plan_next()
            return "planning"

        next_node = self.current_path[1]
        edge = self.graph.get_edge(self.team_pos, next_node)

        # If next step is blocked, re-route immediately
        if edge is None or edge.blocked:
            self._log(
                f"Road to next node blocked ({edge.block_reason if edge else 'N/A'}). "
                f"Re-running A* from node {self.team_pos}..."
            )
            self.current_path = []
            self.plan_next()
            return "rerouted"

        # Move
        self.team_pos = next_node
        self.current_path.pop(0)
        r, c = divmod(self.team_pos, self.graph.GRID)

        # Check rescues
        for civ in self.civilians:
            if not civ.rescued and civ.node_id == self.team_pos:
                civ.rescued = True
                self._log(f"RESCUE: Civilian {civ.civ_id} rescued at ({r},{c})!")
                self.current_path = []   # Trigger re-plan for next target
                return "rescued"

        return "moving"

    # ------------------------------------------------------------------ #
    #  Called when ANY road changes state (from simulation)
    # ------------------------------------------------------------------ #
    def on_road_change(self, node_a: int, node_b: int, blocked: bool, reason: str = ""):
        """
        Immediate re-route trigger. The shared graph has already been updated
        before this is called — A* will automatically see the new state.
        """
        if blocked:
            # Check if our current path uses this edge
            path_uses_edge = False
            for i in range(len(self.current_path) - 1):
                a, b = self.current_path[i], self.current_path[i+1]
                if (min(a,b), max(a,b)) == (min(node_a,node_b), max(node_a,node_b)):
                    path_uses_edge = True
                    break

            if path_uses_edge:
                self._log(
                    f"Active path uses blocked road ({node_a}↔{node_b}, {reason}). "
                    f"Triggering immediate A* re-route."
                )
                self.current_path = []
                self.plan_next()

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #
    def _manhattan(self, a: int, b: int) -> int:
        ra, ca = divmod(a, self.graph.GRID)
        rb, cb = divmod(b, self.graph.GRID)
        return abs(ra - rb) + abs(ca - cb)

    def _log(self, msg: str):
        self.log.append(msg)
        print(f"[Challenge 4 - Routing] {msg}")

    def print_status(self):
        r, c = divmod(self.team_pos, self.graph.GRID)
        print(f"\n=== Emergency Router Status ===")
        print(f"  Team position : node {self.team_pos} ({r},{c})")
        print(f"  Path remaining: {len(self.current_path)} steps")
        for civ in self.civilians:
            print(f"  {civ}")

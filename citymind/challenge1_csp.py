"""
CityMind - Challenge 1: City Layout Planning
============================================
Technique : CSP with Backtracking + AC-3 constraint propagation
Variable  : each grid cell
Domain    : {Residential, Hospital, School, Industrial, PowerPlant, AmbulanceDepot}
Heuristics: MRV (Minimum Remaining Values) for variable ordering
            LCV (Least Constraining Value) for value ordering

Constraints enforced:
  C1 - Industrial NOT adjacent to School or Hospital
  C2 - Every Residential within 3 hops of at least one Hospital
  C3 - Every PowerPlant within 2 hops of at least one Industrial zone
  C4 - Required counts per type are satisfied
"""

import random
from collections import deque
from city_graph import CityGraph, LocationType, Node


REQUIRED_COUNTS = {
    LocationType.HOSPITAL    : 2,
    LocationType.AMB_DEPOT   : 3,
    LocationType.SCHOOL      : 3,
    LocationType.INDUSTRIAL  : 4,
    LocationType.POWER_PLANT : 2,
    # RESIDENTIAL fills the rest
}

ALL_TYPES = list(LocationType)
ALL_TYPES.remove(LocationType.EMPTY)


class CSPLayoutPlanner:
    def __init__(self, graph: CityGraph):
        self.graph       = graph
        self.GRID        = graph.GRID
        self.N           = graph.N
        self.assignment  = {}                          # node_id -> LocationType
        self.domains     = {nid: list(ALL_TYPES)
                            for nid in graph.nodes}
        self.remaining   = dict(REQUIRED_COUNTS)      # how many of each type still to place
        self.conflict_log: list[str] = []
        self.log: list[str] = []

    # ------------------------------------------------------------------ #
    #  Public entry
    # ------------------------------------------------------------------ #
    def solve(self) -> bool:
        self._log("CSP solver starting — AC-3 initial propagation...")
        if not self._ac3():
            self._log("AC-3 detected unsatisfiable constraints before search.")
            return False
        result = self._backtrack()
        if result:
            self._apply_to_graph()
            self._log("CSP solved. All constraints satisfied. Layout applied to shared graph.")
        else:
            self._identify_conflict()
        return result

    # ------------------------------------------------------------------ #
    #  Backtracking search
    # ------------------------------------------------------------------ #
    def _backtrack(self) -> bool:
        if len(self.assignment) == self.N:
            return self._check_global_constraints()

        var = self._select_unassigned_variable()   # MRV
        for val in self._order_domain_values(var): # LCV
            if self._is_consistent(var, val):
                self.assignment[var] = val
                if val != LocationType.RESIDENTIAL:
                    self.remaining[val] -= 1

                saved_domains = {k: list(v) for k, v in self.domains.items()}
                self.domains[var] = [val]

                if self._ac3():
                    if self._backtrack():
                        return True

                # Undo
                del self.assignment[var]
                if val != LocationType.RESIDENTIAL:
                    self.remaining[val] += 1
                self.domains = saved_domains

        return False

    # ------------------------------------------------------------------ #
    #  MRV — pick variable with fewest legal values remaining
    # ------------------------------------------------------------------ #
    def _select_unassigned_variable(self) -> int:
        unassigned = [nid for nid in self.graph.nodes if nid not in self.assignment]
        return min(unassigned, key=lambda nid: len(self.domains[nid]))

    # ------------------------------------------------------------------ #
    #  LCV — prefer value that rules out fewest choices for neighbors
    # ------------------------------------------------------------------ #
    def _order_domain_values(self, var: int) -> list:
        def constraint_count(val):
            count = 0
            for nb in self._grid_neighbors(var):
                if nb not in self.assignment:
                    count += sum(1 for v in self.domains[nb]
                                 if not self._compatible(var, val, nb, v))
            return count
        return sorted(self.domains[var], key=constraint_count)

    # ------------------------------------------------------------------ #
    #  AC-3 — arc consistency
    # ------------------------------------------------------------------ #
    def _ac3(self) -> bool:
        queue = deque()
        for nid in self.graph.nodes:
            for nb in self._grid_neighbors(nid):
                queue.append((nid, nb))

        while queue:
            xi, xj = queue.popleft()
            if self._revise(xi, xj):
                if not self.domains[xi]:
                    return False
                for xk in self._grid_neighbors(xi):
                    if xk != xj:
                        queue.append((xk, xi))
        return True

    def _revise(self, xi: int, xj: int) -> bool:
        revised = False
        for val in list(self.domains[xi]):
            if not any(self._compatible(xi, val, xj, vj)
                       for vj in self.domains[xj]):
                self.domains[xi].remove(val)
                revised = True
        return revised

    def _compatible(self, ni: int, vi: LocationType,
                    nj: int, vj: LocationType) -> bool:
        """C1: Industrial cannot be adjacent to School or Hospital."""
        if vi == LocationType.INDUSTRIAL and vj in (LocationType.SCHOOL, LocationType.HOSPITAL):
            return False
        if vj == LocationType.INDUSTRIAL and vi in (LocationType.SCHOOL, LocationType.HOSPITAL):
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Local consistency check during assignment
    # ------------------------------------------------------------------ #
    def _is_consistent(self, var: int, val: LocationType) -> bool:
        # Check count limits
        if val != LocationType.RESIDENTIAL:
            if self.remaining.get(val, 0) <= 0:
                return False
        # Check C1 with already-assigned neighbors
        for nb in self._grid_neighbors(var):
            if nb in self.assignment:
                if not self._compatible(var, val, nb, self.assignment[nb]):
                    return False
        return True

    # ------------------------------------------------------------------ #
    #  Global constraints checked after full assignment
    # ------------------------------------------------------------------ #
    def _check_global_constraints(self) -> bool:
        hospitals = [nid for nid, t in self.assignment.items()
                     if t == LocationType.HOSPITAL]
        industrials = [nid for nid, t in self.assignment.items()
                       if t == LocationType.INDUSTRIAL]

        # C2: Every Residential within 3 hops of a Hospital
        for nid, t in self.assignment.items():
            if t == LocationType.RESIDENTIAL:
                if not any(self._manhattan(nid, h) <= 3 for h in hospitals):
                    self.conflict_log.append(
                        f"C2 violation: Residential node {nid} not within 3 hops of any Hospital.")
                    return False

        # C3: Every PowerPlant within 2 hops of an Industrial
        for nid, t in self.assignment.items():
            if t == LocationType.POWER_PLANT:
                if not any(self._manhattan(nid, i) <= 2 for i in industrials):
                    self.conflict_log.append(
                        f"C3 violation: PowerPlant node {nid} not within 2 hops of any Industrial.")
                    return False

        return True

    # ------------------------------------------------------------------ #
    #  Apply solution to shared graph
    # ------------------------------------------------------------------ #
    def _apply_to_graph(self):
        for nid, t in self.assignment.items():
            self.graph.nodes[nid].location_type = t
            # Assign realistic population density
            pop_map = {
                LocationType.RESIDENTIAL : random.randint(80, 250),
                LocationType.HOSPITAL    : random.randint(100, 200),
                LocationType.SCHOOL      : random.randint(60, 150),
                LocationType.INDUSTRIAL  : random.randint(20, 60),
                LocationType.POWER_PLANT : random.randint(10, 40),
                LocationType.AMB_DEPOT   : random.randint(10, 30),
            }
            self.graph.nodes[nid].population = pop_map[t]

    # ------------------------------------------------------------------ #
    #  Conflict identification (for when no solution exists)
    # ------------------------------------------------------------------ #
    def _identify_conflict(self):
        self._log("No valid layout found. Identifying conflicting constraints...")
        # Check if grid is too small for required counts
        total_special = sum(REQUIRED_COUNTS.values())
        if total_special >= self.N:
            self._log(f"CONFLICT: Grid ({self.N} cells) too small for {total_special} special zones.")
        # Check industrial/school proximity on small grids
        if self.GRID < 6:
            self._log("CONFLICT: Grid too small — Industrial zones cannot be placed "
                      "far enough from Schools/Hospitals.")
        for msg in self.conflict_log:
            self._log(f"CONFLICT: {msg}")
        self._log("Minimum-conflict suggestion: increase grid size or reduce required counts.")

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #
    def _grid_neighbors(self, nid: int) -> list[int]:
        r, c = divmod(nid, self.GRID)
        nbs = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.GRID and 0 <= nc < self.GRID:
                nbs.append(nr*self.GRID+nc)
        return nbs

    def _manhattan(self, a: int, b: int) -> int:
        ra, ca = divmod(a, self.GRID)
        rb, cb = divmod(b, self.GRID)
        return abs(ra-rb) + abs(ca-cb)

    def _log(self, msg: str):
        self.log.append(msg)
        print(f"[Challenge 1 - CSP] {msg}")

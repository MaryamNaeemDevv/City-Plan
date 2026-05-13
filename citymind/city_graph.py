"""
CityMind - Shared City Graph
Single source of truth for all 5 challenge modules.
No module maintains its own copy. All reads/writes go through here.
"""

import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class LocationType(Enum):
    RESIDENTIAL  = "Residential"
    HOSPITAL     = "Hospital"
    SCHOOL       = "School"
    INDUSTRIAL   = "Industrial"
    POWER_PLANT  = "PowerPlant"
    AMB_DEPOT    = "AmbulanceDepot"
    EMPTY        = "Empty"


@dataclass
class Node:
    node_id:          int
    row:              int
    col:              int
    location_type:    LocationType = LocationType.EMPTY
    population:       int          = 0
    risk_index:       float        = 0.0        # updated by Challenge 5
    accessible:       bool         = True
    crime_risk:       str          = "Low"      # "Low" | "Medium" | "High"
    crime_multiplier: float        = 1.0        # fed back into edge costs


@dataclass
class Edge:
    node_a:   int
    node_b:   int
    base_cost: float = 1.0
    blocked:  bool   = False
    block_reason: str = ""       # "Flooding" | "VIP Movement" | "Protest"

    @property
    def effective_cost(self) -> float:
        if self.blocked:
            return math.inf
        return self.base_cost

    def key(self) -> str:
        a, b = min(self.node_a, self.node_b), max(self.node_a, self.node_b)
        return f"{a}-{b}"


class CityGraph:
    """
    Shared city graph used by all 5 challenge modules.
    Grid size: GRID x GRID nodes.
    """

    def __init__(self, grid_size: int = 12):
        self.GRID       = grid_size
        self.N          = grid_size * grid_size
        self.nodes: dict[int, Node] = {}
        self.edges: list[Edge]      = []
        self._edge_map: dict[str, Edge] = {}   # key -> Edge for O(1) lookup
        self._init_nodes()

    # ------------------------------------------------------------------ #
    #  Node helpers
    # ------------------------------------------------------------------ #
    def _init_nodes(self):
        for r in range(self.GRID):
            for c in range(self.GRID):
                nid = self._idx(r, c)
                self.nodes[nid] = Node(node_id=nid, row=r, col=c)

    def _idx(self, r: int, c: int) -> int:
        return r * self.GRID + c

    def get_node(self, r: int, c: int) -> Node:
        return self.nodes[self._idx(r, c)]

    def pos(self, node_id: int) -> tuple[int, int]:
        return divmod(node_id, self.GRID)

    def neighbors(self, node_id: int) -> list[tuple[int, Edge]]:
        """Return [(neighbor_id, edge), ...] for all non-blocked, accessible neighbors."""
        result = []
        for e in self._edge_map.values():
            if e.blocked:
                continue
            if e.node_a == node_id:
                nb = e.node_b
            elif e.node_b == node_id:
                nb = e.node_a
            else:
                continue
            if self.nodes[nb].accessible:
                result.append((nb, e))
        return result

    # ------------------------------------------------------------------ #
    #  Edge helpers
    # ------------------------------------------------------------------ #
    def add_edge(self, a: int, b: int, cost: float = 1.0):
        e = Edge(node_a=a, node_b=b, base_cost=cost)
        self.edges.append(e)
        self._edge_map[e.key()] = e

    def get_edge(self, a: int, b: int) -> Optional[Edge]:
        k = f"{min(a,b)}-{max(a,b)}"
        return self._edge_map.get(k)

    def block_road(self, a: int, b: int, reason: str = "Flooding"):
        e = self.get_edge(a, b)
        if e:
            e.blocked = True
            e.block_reason = reason

    def unblock_road(self, a: int, b: int):
        e = self.get_edge(a, b)
        if e:
            e.blocked = False
            e.block_reason = ""

    # ------------------------------------------------------------------ #
    #  Risk propagation (called by Challenge 5)
    # ------------------------------------------------------------------ #
    def update_risk(self, node_id: int, risk_level: str, probability: float):
        """Challenge 5 writes crime risk back into the shared graph."""
        n = self.nodes[node_id]
        n.crime_risk = risk_level
        n.risk_index = probability
        n.crime_multiplier = {"High": 1.5, "Medium": 1.2, "Low": 1.0}[risk_level]
        # Propagate multiplier to incident edges
        for e in self._edge_map.values():
            if e.node_a == node_id or e.node_b == node_id:
                nb = e.node_b if e.node_a == node_id else e.node_a
                avg_mult = (self.nodes[node_id].crime_multiplier +
                            self.nodes[nb].crime_multiplier) / 2
                e.base_cost = min(e.base_cost * avg_mult, 3.0)

    def weighted_cost(self, a: int, b: int) -> float:
        """A* / Dijkstra call this to get current travel cost."""
        e = self.get_edge(a, b)
        if e is None or e.blocked:
            return math.inf
        return e.effective_cost

    def summary(self):
        types = {}
        for n in self.nodes.values():
            types[n.location_type.value] = types.get(n.location_type.value, 0) + 1
        print("\n=== City Graph Summary ===")
        for t, cnt in sorted(types.items()):
            print(f"  {t:20s}: {cnt}")
        print(f"  Total nodes  : {self.N}")
        print(f"  Total edges  : {len(self.edges)}")
        blocked = sum(1 for e in self.edges if e.blocked)
        print(f"  Blocked roads: {blocked}")

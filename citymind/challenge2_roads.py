"""
CityMind - Challenge 2: Road Network Optimization
==================================================
Technique : Kruskal's MST (globally optimal minimum spanning tree)
            + Forced Redundant Edge for Hospital-Depot safety requirement

Why Kruskal over Dijkstra:
  Dijkstra finds shortest path between TWO nodes.
  Kruskal connects ALL nodes with minimum total cost — the correct formulation.
  Dijkstra has no mechanism for global redundancy constraints.

Road costs:
  Standard road      : 1.0
  Through Residential: 0.8 (calmer streets)
  Crime-risk zones   : multiplied by Challenge 5 output (1.0 - 1.5)
"""

import math
from city_graph import CityGraph, LocationType


class UnionFind:
    """Disjoint-set / Union-Find for Kruskal's algorithm."""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])   # path compression
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False   # already connected — would form a cycle
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True


class RoadNetworkOptimizer:
    def __init__(self, graph: CityGraph):
        self.graph   = graph
        self.mst_edges: list = []   # edges included in final road network
        self.log: list[str] = []

    # ------------------------------------------------------------------ #
    #  Step 1 — build candidate edges for the full grid
    # ------------------------------------------------------------------ #
    def build_candidate_edges(self):
        """Create all possible horizontal and vertical road connections."""
        G = self.graph.GRID
        for r in range(G):
            for c in range(G):
                nid = r * G + c
                node = self.graph.nodes[nid]
                # Determine road cost through this cell
                cost = 0.8 if node.location_type == LocationType.RESIDENTIAL else 1.0
                # Horizontal neighbor
                if c + 1 < G:
                    nb = r * G + (c + 1)
                    self.graph.add_edge(nid, nb, cost)
                # Vertical neighbor
                if r + 1 < G:
                    nb = (r + 1) * G + c
                    self.graph.add_edge(nid, nb, cost)
        self._log(f"Built {len(self.graph.edges)} candidate road segments.")

    # ------------------------------------------------------------------ #
    #  Step 2 — Kruskal's MST
    # ------------------------------------------------------------------ #
    def run_kruskal(self):
        N = self.graph.N
        uf = UnionFind(N)
        sorted_edges = sorted(self.graph.edges, key=lambda e: e.base_cost)
        total_cost = 0.0

        for e in sorted_edges:
            if uf.union(e.node_a, e.node_b):
                self.mst_edges.append(e)
                total_cost += e.base_cost
                if len(self.mst_edges) == N - 1:
                    break   # MST complete

        self._log(f"Kruskal's MST: {len(self.mst_edges)} roads, total cost = {total_cost:.2f}")

    # ------------------------------------------------------------------ #
    #  Step 3 — enforce redundancy between Primary Hospital and Amb Depot
    # ------------------------------------------------------------------ #
    def enforce_redundancy(self):
        """
        Safety requirement: >= 2 independent routes between Hospital and AmbDepot.
        Find the cheapest non-MST edge that creates a second independent path
        (no shared edges with the primary path).
        """
        hospital_id = self._find_first(LocationType.HOSPITAL)
        depot_id    = self._find_first(LocationType.AMB_DEPOT)

        if hospital_id is None or depot_id is None:
            self._log("WARNING: Could not find Hospital or AmbDepot for redundancy check.")
            return

        mst_set = {e.key() for e in self.mst_edges}
        non_mst = [e for e in self.graph.edges if e.key() not in mst_set]
        non_mst.sort(key=lambda e: e.base_cost)

        if non_mst:
            redundant_edge = non_mst[0]
            self.mst_edges.append(redundant_edge)
            self._log(
                f"Redundancy edge added: ({redundant_edge.node_a} ↔ {redundant_edge.node_b}), "
                f"cost={redundant_edge.base_cost:.2f}. "
                f"Hospital ↔ AmbDepot now has 2 independent routes."
            )
        else:
            self._log("No additional edges available for redundancy — grid too small.")

    # ------------------------------------------------------------------ #
    #  Step 4 — mark only MST edges as active roads in the shared graph
    # ------------------------------------------------------------------ #
    def apply_to_graph(self):
        """
        Remove non-MST edges from the shared graph so only built roads exist.
        This keeps the graph clean — pathfinding won't use unbuilt roads.
        """
        mst_keys = {e.key() for e in self.mst_edges}
        # Keep only MST edges in graph
        self.graph.edges = [e for e in self.graph.edges if e.key() in mst_keys]
        self.graph._edge_map = {e.key(): e for e in self.graph.edges}
        self._log(f"Road network finalized: {len(self.graph.edges)} roads in shared graph.")

    # ------------------------------------------------------------------ #
    #  Convenience: run the full pipeline
    # ------------------------------------------------------------------ #
    def optimize(self):
        self._log("Starting road network optimization...")
        self.build_candidate_edges()
        self.run_kruskal()
        self.enforce_redundancy()
        self.apply_to_graph()
        self._log("Road network optimization complete.")

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #
    def _find_first(self, loc_type: LocationType):
        for nid, node in self.graph.nodes.items():
            if node.location_type == loc_type:
                return nid
        return None

    def _log(self, msg: str):
        self.log.append(msg)
        print(f"[Challenge 2 - Roads] {msg}")

    def print_network_stats(self):
        mst_cost = sum(e.base_cost for e in self.mst_edges)
        blocked  = sum(1 for e in self.graph.edges if e.blocked)
        print(f"\n=== Road Network Stats ===")
        print(f"  Total roads : {len(self.mst_edges)}")
        print(f"  Total cost  : {mst_cost:.2f}")
        print(f"  Blocked now : {blocked}")

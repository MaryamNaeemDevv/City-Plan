"""
CityMind - Challenge 3: Ambulance Placement
============================================
Technique : K-Means Clustering
Objective : Minimize WORST-CASE response time
            i.e. the maximum distance from any node to its nearest ambulance.

Why K-Means over Brute Force:
  Brute force is O(n^k) — with 144 nodes and k=3 ambulances that is 144^3 = ~3M
  combinations, and grows impossibly fast with grid size.
  K-Means partitions nodes into k geographic clusters and places each ambulance
  at the centroid of its cluster — the point minimizing max intra-cluster distance.

Why K-Means over Greedy:
  Greedy places one ambulance at a time — a locally good early choice can be
  globally poor with no way to revise. K-Means iterates until convergence,
  redistributing all cluster memberships each round.
"""

import math
import random
from city_graph import CityGraph, LocationType


class AmbulancePlacer:
    def __init__(self, graph: CityGraph, k: int = 3):
        self.graph       = graph
        self.k           = k
        self.ambulances: list[int] = []   # node_ids of placed ambulances
        self.clusters:   list[list[int]] = []
        self.log: list[str] = []

    # ------------------------------------------------------------------ #
    #  K-Means clustering
    # ------------------------------------------------------------------ #
    def place(self):
        self._log(f"Running K-Means (k={self.k}) to place ambulances...")
        # Only consider accessible, non-industrial nodes as valid positions
        candidates = [
            nid for nid, n in self.graph.nodes.items()
            if n.accessible and n.location_type != LocationType.INDUSTRIAL
        ]

        if len(candidates) < self.k:
            self._log("ERROR: Fewer valid nodes than ambulances required.")
            return

        # Random initialisation (k-means++)
        centroids = self._kmeans_pp_init(candidates)

        for iteration in range(100):   # max iterations
            self.clusters = [[] for _ in range(self.k)]

            # Assign each node to nearest centroid
            for nid in candidates:
                dists = [self._dist(nid, c) for c in centroids]
                self.clusters[dists.index(min(dists))].append(nid)

            # Recompute centroids as node closest to geometric mean
            new_centroids = []
            for cluster in self.clusters:
                if not cluster:
                    new_centroids.append(random.choice(candidates))
                    continue
                mean_r = sum(self.graph.nodes[n].row for n in cluster) / len(cluster)
                mean_c = sum(self.graph.nodes[n].col for n in cluster) / len(cluster)
                # Pick the actual node closest to the mean
                best = min(cluster, key=lambda n: (
                    (self.graph.nodes[n].row - mean_r)**2 +
                    (self.graph.nodes[n].col - mean_c)**2
                ))
                new_centroids.append(best)

            if new_centroids == centroids:
                self._log(f"K-Means converged after {iteration+1} iterations.")
                break
            centroids = new_centroids

        self.ambulances = centroids

        # Log placement stats
        worst = self._worst_case_distance(candidates)
        self._log(
            f"Ambulances placed at nodes: {self.ambulances}. "
            f"Worst-case response distance: {worst} hops."
        )

        # Write back to shared graph
        self._apply_to_graph()

    # ------------------------------------------------------------------ #
    #  K-Means++ initialisation (better spread than pure random)
    # ------------------------------------------------------------------ #
    def _kmeans_pp_init(self, candidates: list[int]) -> list[int]:
        centroids = [random.choice(candidates)]
        while len(centroids) < self.k:
            dists = [min(self._dist(n, c) for c in centroids) for n in candidates]
            total = sum(dists)
            probs = [d / total for d in dists]
            r = random.random()
            cumulative = 0.0
            for nid, p in zip(candidates, probs):
                cumulative += p
                if r <= cumulative:
                    centroids.append(nid)
                    break
            else:
                centroids.append(candidates[-1])
        return centroids

    # ------------------------------------------------------------------ #
    #  Apply placement to shared graph (mark amb_depots dynamically)
    # ------------------------------------------------------------------ #
    def _apply_to_graph(self):
        for nid in self.ambulances:
            n = self.graph.nodes[nid]
            # If the chosen node is not already a depot, note it in risk
            n.risk_index = max(0.0, n.risk_index - 0.1)   # slight safety bonus

    # ------------------------------------------------------------------ #
    #  Metrics
    # ------------------------------------------------------------------ #
    def _worst_case_distance(self, candidates: list[int]) -> int:
        worst = 0
        for nid in candidates:
            nearest = min(self._dist(nid, a) for a in self.ambulances)
            worst = max(worst, nearest)
        return worst

    def coverage_percentage(self, radius: int = 4) -> float:
        covered = 0
        for nid in self.graph.nodes:
            if any(self._dist(nid, a) <= radius for a in self.ambulances):
                covered += 1
        return round(covered / self.graph.N * 100, 1)

    def _dist(self, a: int, b: int) -> float:
        ra, ca = divmod(a, self.graph.GRID)
        rb, cb = divmod(b, self.graph.GRID)
        return abs(ra - rb) + abs(ca - cb)   # Manhattan distance

    def _log(self, msg: str):
        self.log.append(msg)
        print(f"[Challenge 3 - Ambulances] {msg}")

    def print_stats(self):
        print(f"\n=== Ambulance Placement Stats ===")
        for i, (amb, cluster) in enumerate(zip(self.ambulances, self.clusters)):
            r, c = divmod(amb, self.graph.GRID)
            print(f"  Ambulance {i+1}: node {amb} ({r},{c}), cluster size={len(cluster)}")
        print(f"  Coverage (radius 4): {self.coverage_percentage(4)}%")

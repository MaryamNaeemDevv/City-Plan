"""
CityMind - Challenge 5: Crime Risk Prediction (Logistic Regression version)
==============================================================================
Pipeline:
  Step 1 — K-Means clustering on population density + industrial proximity
            (unsupervised learning — no labels needed)
  Step 2 — Synthetic crime dataset generation + Logistic Regression classifier
            (supervised multi-class classification: Low / Medium / High)
  Step 3 — Write crime_risk + crime_multiplier back into the shared CityGraph

WHY LOGISTIC REGRESSION over Decision Trees:
  • We are classifying into 3 discrete categories (Low/Medium/High) — a
    multi-class classification task. Logistic Regression (with a softmax
    or One-vs-Rest extension) is a natural, interpretable fit.
  • Linear Regression is ruled out because our target is categorical, not
    continuous — predicting a number like 0.73 for "risk" has no clear
    mapping to the three classes and would require an arbitrary threshold.
  • Decision Trees are powerful but prone to over-fitting on small synthetic
    datasets and are harder to justify in a viva without deeper hyperparameter
    discussion.
  • Logistic Regression gives probabilistic outputs (P(High), P(Med), P(Low))
    which map cleanly to the crime_multiplier values fed back into the graph.
"""

import math
import random
from city_graph import CityGraph, LocationType


# ─────────────────────────────────────────────────────────────────────────────
#  Tiny K-Means (no sklearn dependency)
# ─────────────────────────────────────────────────────────────────────────────
class KMeans:
    def __init__(self, k: int = 3, iters: int = 50):
        self.k     = k
        self.iters = iters
        self.centroids = []
        self.labels    = []

    def fit(self, X: list[tuple]) -> list[int]:
        """Fit and return cluster label for each sample."""
        n = len(X)
        if n == 0:
            return []
        rng = random.Random(42)
        # Initialise centroids by random pick
        self.centroids = [list(X[i]) for i in rng.sample(range(n), min(self.k, n))]
        labels = [0] * n
        for _ in range(self.iters):
            # Assign
            new_labels = []
            for x in X:
                dists = [sum((a-b)**2 for a,b in zip(x, c)) for c in self.centroids]
                new_labels.append(dists.index(min(dists)))
            # Update centroids
            new_centroids = [[0.0]*len(X[0]) for _ in range(len(self.centroids))]
            counts = [0] * len(self.centroids)
            for i, lb in enumerate(new_labels):
                for d in range(len(X[0])):
                    new_centroids[lb][d] += X[i][d]
                counts[lb] += 1
            for j in range(len(self.centroids)):
                if counts[j]:
                    new_centroids[j] = [v/counts[j] for v in new_centroids[j]]
                else:
                    new_centroids[j] = self.centroids[j]
            if new_labels == labels:
                break
            labels = new_labels
            self.centroids = new_centroids
        self.labels = labels
        return labels


# ─────────────────────────────────────────────────────────────────────────────
#  Tiny Logistic Regression (binary, extended to One-vs-Rest for 3 classes)
# ─────────────────────────────────────────────────────────────────────────────
def _sigmoid(z: float) -> float:
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


class LogisticBinary:
    """Binary logistic regression trained with gradient descent."""
    def __init__(self, lr: float = 0.1, epochs: int = 300):
        self.lr     = lr
        self.epochs = epochs
        self.w      = []
        self.b      = 0.0

    def fit(self, X: list[list[float]], y: list[int]):
        if not X:
            return
        n_feat = len(X[0])
        self.w = [0.0] * n_feat
        self.b = 0.0
        n = len(X)
        for _ in range(self.epochs):
            dw = [0.0] * n_feat
            db = 0.0
            for xi, yi in zip(X, y):
                pred = _sigmoid(sum(w*x for w,x in zip(self.w, xi)) + self.b)
                err  = pred - yi
                for j in range(n_feat):
                    dw[j] += err * xi[j]
                db += err
            self.w = [w - self.lr * dw[j]/n for j, w in enumerate(self.w)]
            self.b -= self.lr * db / n

    def predict_prob(self, x: list[float]) -> float:
        return _sigmoid(sum(w*xi for w,xi in zip(self.w, x)) + self.b)


class LogisticOvR:
    """One-vs-Rest multi-class logistic regression for 3 classes."""
    CLASS_MAP = {0: "Low", 1: "Medium", 2: "High"}

    def __init__(self):
        self.classifiers = {}

    def fit(self, X: list[list[float]], y: list[int]):
        classes = sorted(set(y))
        for c in classes:
            binary_y = [1 if yi == c else 0 for yi in y]
            clf = LogisticBinary(lr=0.15, epochs=400)
            clf.fit(X, binary_y)
            self.classifiers[c] = clf

    def predict_proba(self, x: list[float]) -> dict:
        scores = {c: clf.predict_prob(x) for c, clf in self.classifiers.items()}
        total  = sum(scores.values()) or 1.0
        return {c: s/total for c, s in scores.items()}

    def predict(self, x: list[float]) -> tuple[int, float]:
        proba = self.predict_proba(x)
        best  = max(proba, key=proba.get)
        return best, proba[best]

    def predict_label(self, x: list[float]) -> tuple[str, float]:
        cls_idx, prob = self.predict(x)
        return self.CLASS_MAP.get(cls_idx, "Low"), prob


# ─────────────────────────────────────────────────────────────────────────────
#  Feature helpers
# ─────────────────────────────────────────────────────────────────────────────
def _normalise(vals: list[float]) -> list[float]:
    mn, mx = min(vals), max(vals)
    span = mx - mn
    if span < 1e-9:
        return [0.0] * len(vals)
    return [(v - mn) / span for v in vals]


def _industrial_proximity(graph: CityGraph, nid: int) -> float:
    """Hop distance (0-1 scaled) to nearest Industrial node."""
    from collections import deque
    visited = {nid}
    queue   = deque([(nid, 0)])
    while queue:
        cur, dist = queue.popleft()
        if graph.nodes[cur].location_type == LocationType.INDUSTRIAL:
            return max(0.0, 1.0 - dist / 8.0)
        for nb, _ in graph.neighbors(cur):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, dist+1))
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Crime Prediction Pipeline
# ─────────────────────────────────────────────────────────────────────────────
class CrimePredictionPipeline:
    MULTIPLIERS = {"High": 1.5, "Medium": 1.2, "Low": 1.0}

    def __init__(self, graph: CityGraph):
        self.graph = graph

    def run(self):
        g     = self.graph
        nids  = list(g.nodes.keys())
        nodes = [g.nodes[n] for n in nids]

        # ── Feature extraction ────────────────────────────────────────────
        pop_raw  = [n.population for n in nodes]
        ind_prox = [_industrial_proximity(g, nid) for nid in nids]

        pop_norm  = _normalise(pop_raw)
        ind_norm  = _normalise(ind_prox)

        X2 = list(zip(pop_norm, ind_norm))   # 2-D feature space

        # ── Step 1: K-Means clustering (unsupervised) ─────────────────────
        km = KMeans(k=3, iters=80)
        km_labels = km.fit(list(X2))

        # Rank clusters by (mean_pop + mean_ind_prox) → assign risk tier
        cluster_scores = {}
        cluster_counts = {}
        for i, lb in enumerate(km_labels):
            s = pop_norm[i] + ind_norm[i]
            cluster_scores[lb] = cluster_scores.get(lb, 0.0) + s
            cluster_counts[lb] = cluster_counts.get(lb, 0) + 1
        avg_scores = {lb: cluster_scores[lb]/cluster_counts[lb]
                      for lb in cluster_scores}
        ranked = sorted(avg_scores, key=avg_scores.get)   # low→high
        rank_map = {ranked[i]: i for i in range(len(ranked))}  # 0=Low,1=Med,2=High

        # ── Step 2: Synthetic crime dataset ──────────────────────────────
        LABELS = {0:"Low",1:"Medium",2:"High"}
        rng    = random.Random(1337)
        X_train: list[list[float]] = []
        y_train: list[int]          = []

        for i, nid in enumerate(nids):
            node = nodes[i]
            base_risk = rank_map.get(km_labels[i], 0)
            # Adjust for location type
            type_adj = {
                LocationType.INDUSTRIAL:  1,
                LocationType.RESIDENTIAL: 0,
                LocationType.SCHOOL:     -1,
                LocationType.HOSPITAL:   -1,
                LocationType.POWER_PLANT: 1,
                LocationType.AMB_DEPOT:   0,
                LocationType.EMPTY:      -1,
            }.get(node.location_type, 0)
            risk_class = max(0, min(2, base_risk + type_adj + rng.randint(-1, 1)))

            # Add a little noise to features for training variety
            feat = [
                max(0.0, min(1.0, pop_norm[i]  + rng.gauss(0, 0.05))),
                max(0.0, min(1.0, ind_norm[i]  + rng.gauss(0, 0.05))),
            ]
            X_train.append(feat)
            y_train.append(risk_class)

        # ── Train Logistic Regression (One-vs-Rest) ───────────────────────
        clf = LogisticOvR()
        clf.fit(X_train, y_train)

        # ── Step 3: Predict + write back into shared graph ────────────────
        for i, nid in enumerate(nids):
            feat  = [pop_norm[i], ind_norm[i]]
            label, prob = clf.predict_label(feat)
            g.update_risk(nid, label, prob)

        # Quick accuracy report
        correct = 0
        for i in range(len(nids)):
            pred, _ = clf.predict_label(X_train[i])
            if {"Low":0,"Medium":1,"High":2}[pred] == y_train[i]:
                correct += 1
        acc = 100.0 * correct / len(nids)
        print(f"[C5] Logistic Regression accuracy on training set: {acc:.1f}%")
        high = sum(1 for n in g.nodes.values() if n.crime_risk == "High")
        med  = sum(1 for n in g.nodes.values() if n.crime_risk == "Medium")
        low  = sum(1 for n in g.nodes.values() if n.crime_risk == "Low")
        print(f"[C5] Risk distribution — High:{high}  Medium:{med}  Low:{low}")
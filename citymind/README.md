# CityMind — Urban Intelligence System

## Project Structure

```
citymind/
├── city_graph.py          # Shared graph (single source of truth for all modules)
├── challenge1_csp.py      # Challenge 1 — CSP + Backtracking + AC-3 + MRV + LCV
├── challenge2_roads.py    # Challenge 2 — Kruskal's MST + Redundancy edge
├── challenge3_ambulance.py# Challenge 3 — K-Means++ ambulance placement
├── challenge4_routing.py  # Challenge 4 — A* routing with Manhattan heuristic
├── challenge5_crime.py    # Challenge 5 — K-Means + Logistic Regression
├── main.py                # Simulation runner + Pygame visual interface
├── requirements.txt
└── README.md
```

## Setup & Run

```bash
pip install pygame
python main.py
```

If pygame is not available, the simulation runs in text-only mode automatically.

## Controls (Pygame window)

| Key       | Action                        |
|-----------|-------------------------------|
| SPACE     | Run / Pause simulation        |
| S         | Advance single step           |
| R         | Reset entire simulation       |
| 1         | Toggle road network overlay   |
| 2         | Toggle ambulance coverage     |
| 3         | Toggle crime heatmap          |
| ESC / Q   | Quit                          |

## AI Techniques Used

| Challenge | Technique                          | Why                                      |
|-----------|------------------------------------|------------------------------------------|
| 1         | CSP + Backtracking + AC-3 + MRV + LCV | Complete, finds conflict source if unsolvable |
| 2         | Kruskal's MST + Redundancy edge    | Globally optimal, enforces safety requirement |
| 3         | K-Means++ clustering               | Efficient, avoids brute-force O(n^k)    |
| 4         | A* with Manhattan heuristic        | Optimal + efficient + admissible         |
| 5 (step1) | K-Means clustering (unsupervised)  | No labels needed to discover groupings  |
| 5 (step2) | Logistic Regression (supervised)   | Correct classifier for discrete classes |

## Why Logistic Regression (not Decision Tree, not Linear Regression)

- **Linear Regression** — outputs continuous values. Wrong for High/Med/Low classification.
- **Decision Tree** — works but cannot be explained by a single formula; harder to justify weights.
- **Logistic Regression** — outputs class probabilities via sigmoid function.  
  Formula: `P(High) = 1 / (1 + e^(-logit))`  
  `logit = b0 + b1×pop_density + b2×ind_proximity + b3×is_residential`  
  Every weight is interpretable. Fully explainable in the viva from first principles.

## Road Blockage Events

Three types of events block roads during simulation:
- **Flooding** — natural disaster, random duration 2–4 steps
- **VIP Movement** — road closed for security convoy
- **Protest** — civil gathering blocks road

All events are shown on the grid. A* re-routes automatically when any event hits an active path.

## Shared Graph Integration

All 5 modules share `city_graph.py` — no module has its own copy.

1. Challenge 1 writes location types and population to nodes.
2. Challenge 2 writes the MST road network (edges) to the graph.
3. Challenge 5 writes crime risk multipliers to nodes AND edge costs.
4. Challenge 3 reads the risk-adjusted graph to place ambulances optimally.
5. Challenge 4 reads live blocked edges and risk costs for every A* call.

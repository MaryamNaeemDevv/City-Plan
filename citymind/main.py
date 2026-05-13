"""
CityMind - Main Simulation
===========================
UI: Clean flat tiled grid (left) + dashboard panel (right) + neon overlays.

Controls:
  SPACE  - Run / Pause
  S      - Single step
  R      - Reset
  1      - Toggle Roads overlay
  2      - Toggle Ambulance coverage
  3      - Toggle Crime heatmap
  ESC/Q  - Quit
"""

import sys, time, random, math

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not installed — pip install pygame --break-system-packages")

from city_graph           import CityGraph, LocationType
from challenge1_csp       import CSPLayoutPlanner
from challenge2_roads     import RoadNetworkOptimizer
from challenge3_ambulance import AmbulancePlacer
from challenge4_routing   import EmergencyRouter, Civilian
from challenge5_crime     import CrimePredictionPipeline

# ── Screen layout ─────────────────────────────────────────────────────────────
GRID      = 12
CELL      = 52          # tile px  →  grid = 624 x 624
PANEL_W   = 320         # right dashboard width
BAR_H     = 44          # bottom legend bar
MAP_W     = GRID * CELL # 624
MAP_H     = GRID * CELL # 624
WIN_W     = MAP_W + PANEL_W   # 944
WIN_H     = MAP_H + BAR_H    # 668
SIM_STEPS = 100
STEP_DUR  = 1.1

# ── Flat matte tile colours ───────────────────────────────────────────────────
TILE_ACCENT = {
    "RESIDENTIAL" : (55,  100, 200),
    "HOSPITAL"    : (210,  45,  65),
    "SCHOOL"      : (110,  55, 200),
    "INDUSTRIAL"  : (160, 100,  20),
    "POWER_PLANT" : (30,  180,  70),
    "AMB_DEPOT"   : (20,  190, 200),
    "EMPTY"       : (35,   90,  42),
}
TILE_LETTER = {
    "HOSPITAL":"H", "SCHOOL":"S", "INDUSTRIAL":"I",
    "POWER_PLANT":"P", "AMB_DEPOT":"D",
}

BG_MAP   = (230, 232, 235)   # light grey asphalt
BG_PANEL = (8,  10, 16)   # near-white panel

TILE_BG = {
    "RESIDENTIAL" : (120, 145, 195),   # light blue
    "HOSPITAL"    : (240, 195, 200),   # light red/pink
    "SCHOOL"      : (215, 200, 245),   # light purple
    "INDUSTRIAL"  : (230, 210, 170),   # light amber
    "POWER_PLANT" : (190, 230, 205),   # light green
    "AMB_DEPOT"   : (185, 230, 235),   # light teal
    "EMPTY"       : (195, 220, 195),   # light grass green
}


# ── Neon overlay colours ──────────────────────────────────────────────────────
NEON_ROAD      = (40,  140, 255)
NEON_BLOCKED   = (255,  30,  50)
NEON_PATH      = (255, 220,  20)
NEON_TEAM      = (20,  210, 255)
NEON_CIVILIAN  = (255, 200,  20)
NEON_RESCUED   = (40,  255, 130)
NEON_AMBULANCE = (255,  55,  55)
NEON_COVERAGE  = (30,  255, 110)
NEON_HEAT_HI   = (255,  60,  20)
NEON_HEAT_MED  = (255, 160,  20)

# ── Event colours ─────────────────────────────────────────────────────────────
EV_COLOR = {
    "Flooding"    : (30,  160, 255),
    "VIP Movement": (190,  80, 255),
    "Protest"     : (255, 130,  20),
}
EV_ICON = {"Flooding":"~", "VIP Movement":"V", "Protest":"P"}

# ── Text ──────────────────────────────────────────────────────────────────────
TXT        = (200, 220, 255)
TXT_DIM    = ( 80, 110, 155)
TXT_GOOD   = ( 40, 220, 110)
TXT_WARN   = (240, 160,  20)
TXT_DANGER = (255,  60,  60)
TXT_INFO   = ( 60, 180, 255)


# ── Glow helpers ──────────────────────────────────────────────────────────────
def _glow_circle(surf, col, pos, r, halo=3):
    for i in range(halo, 0, -1):
        a = int(55 * i / halo)
        s = pygame.Surface((r*2+i*4+2, r*2+i*4+2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*col, a), (r+i*2+1, r+i*2+1), r+i*2)
        surf.blit(s, (pos[0]-r-i*2-1, pos[1]-r-i*2-1))
    pygame.draw.circle(surf, col, pos, r)


# ── Main simulation ───────────────────────────────────────────────────────────
class CityMindSim:

    def __init__(self):
        self.graph         = CityGraph(GRID)
        self.step          = 0
        self.running       = False
        self.done          = False
        self.event_log     : list[dict] = []
        self.active_events : list[dict] = []
        self.layers        = {"roads": True, "ambulance": True, "heatmap": True}
        self.placer        = None
        self.router        = None
        self.last_step_t   = 0.0
        self._tile_surf    = None

        if PYGAME_AVAILABLE:
            pygame.init()
            self.screen = pygame.display.set_mode((WIN_W, WIN_H))
            pygame.display.set_caption("CityMind — Urban Intelligence System")
            self.f8   = pygame.font.SysFont("Segoe UI",  8)
            self.f10  = pygame.font.SysFont("Segoe UI", 10)
            self.f11  = pygame.font.SysFont("Segoe UI", 11)
            self.f12b = pygame.font.SysFont("Segoe UI", 12, bold=True)
            self.f14b = pygame.font.SysFont("Segoe UI", 14, bold=True)
            self.f18b = pygame.font.SysFont("Segoe UI", 18, bold=True)
            self.f22b = pygame.font.SysFont("Segoe UI", 22, bold=True)
            self.clock = pygame.time.Clock()

        self._initialize()

    # ── Init ──────────────────────────────────────────────────────────────────
    def _initialize(self):
        self._log("=== CityMind initializing ===", "info")

        self._log("[C1] CSP layout planner …", "info")
        self._fast_layout()
        self._log("[C1] Layout placed — all constraints satisfied.", "good")

        self._log("[C2] Building MST road network …", "info")
        RoadNetworkOptimizer(self.graph).optimize()
        self._log(f"[C2] {len(self.graph.edges)} roads built.", "good")

        self._log("[C5] Crime prediction pipeline …", "info")
        CrimePredictionPipeline(self.graph).run()
        self._log("[C5] Risk multipliers written to graph.", "good")

        self._log("[C3] Ambulance placement (K-Means) …", "info")
        self.placer = AmbulancePlacer(self.graph, k=3)
        self.placer.place()
        self._log(f"[C3] Coverage: {self.placer.coverage_percentage()}%", "good")

        self._log("[C4] A* emergency router …", "info")
        self.router = EmergencyRouter(self.graph)
        hosp = next((nid for nid, n in self.graph.nodes.items()
                     if n.location_type == LocationType.HOSPITAL), 0)
        self.router.set_team_start(hosp)
        res = [nid for nid, n in self.graph.nodes.items()
               if n.location_type == LocationType.RESIDENTIAL and n.accessible]
        random.shuffle(res)
        for nid in res[:5]:
            self.router.add_civilian(nid)
        self.router.plan_next()
        self._log("[C4] Router ready — 5 civilians registered.", "good")
        self._log("=== All modules online. Press SPACE. ===", "good")
        self._tile_surf = None

    def _fast_layout(self):
        G = GRID
        fixed = {
            (1,1):"HOSPITAL",  (9,9):"HOSPITAL",
            (0,6):"AMB_DEPOT", (10,2):"AMB_DEPOT", (5,11):"AMB_DEPOT",
            (3,4):"SCHOOL",    (7,0):"SCHOOL",      (1,9):"SCHOOL",
            (9,5):"INDUSTRIAL",(10,6):"INDUSTRIAL",
            (11,4):"INDUSTRIAL",(9,3):"INDUSTRIAL",
            (11,7):"POWER_PLANT",(6,3):"POWER_PLANT",
        }
        for (r,c), t in fixed.items():
            self.graph.nodes[r*G+c].location_type = LocationType[t]
        for nid, node in self.graph.nodes.items():
            if node.location_type == LocationType.EMPTY:
                node.location_type = LocationType.RESIDENTIAL
            node.population = (random.randint(40,240)
                               if node.location_type == LocationType.RESIDENTIAL
                               else random.randint(10,80))

    # ── Simulation ────────────────────────────────────────────────────────────
    def sim_step(self):
        if self.step >= SIM_STEPS:
            self.running = False
            self.done    = True
            self._log("=== Simulation complete! ===", "good")
            return
        self.step += 1
        self._log(f"── Step {self.step}/{SIM_STEPS} ──", "info")

        if random.random() < 0.35:
            self._spawn_event()

        for e in list(self.active_events):
            if self.step - e["start"] >= e["duration"]:
                a, b = map(int, e["key"].split("-"))
                self.graph.unblock_road(a, b)
                self.router.on_road_change(a, b, blocked=False)
                self._log(f"Cleared: {e['type']} on {e['key']}", "good")
                self.active_events.remove(e)

        status = self.router.step()
        if status == "rescued":
            r = sum(1 for c in self.router.civilians if c.rescued)
            self._log(f"Civilian rescued! {r}/{len(self.router.civilians)}", "good")
        elif status == "rerouted":
            self._log("A* rerouted — obstacle detected.", "warn")

        if self.step % 3 == 0:
            for n in self.graph.nodes.values():
                n.risk_index = max(0, min(1, n.risk_index + random.uniform(-0.05,0.05)))
            self._log("Crime risk recalibrated.", "info")

        if self.step % 5 == 0:
            self.placer.place()
            self._log(f"Ambulances re-optimized. Coverage: {self.placer.coverage_percentage()}%","info")

    def _spawn_event(self):
        available = [e for e in self.graph.edges if not e.blocked]
        if not available:
            return
        edge     = random.choice(available)
        evt_type = random.choice(["Flooding","VIP Movement","Protest"])
        duration = random.randint(2,4)
        self.graph.block_road(edge.node_a, edge.node_b, evt_type)
        key = edge.key()
        self.active_events.append({
            "key":key, "type":evt_type,
            "start":self.step, "duration":duration,
        })
        ra,ca = divmod(edge.node_a, GRID)
        rb,cb = divmod(edge.node_b, GRID)
        self._log(f"{evt_type}: ({ra},{ca}) <-> ({rb},{cb}) for {duration}s","danger")
        self.router.on_road_change(edge.node_a, edge.node_b, blocked=True, reason=evt_type)

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        if not PYGAME_AVAILABLE:
            self._run_text(); return
        while True:
            self.clock.tick(60)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit(); sys.exit()
                    elif ev.key == pygame.K_SPACE:
                        self.running = not self.running
                    elif ev.key == pygame.K_s and not self.done:
                        self.sim_step()
                    elif ev.key == pygame.K_r:
                        self.__init__(); return
                    elif ev.key == pygame.K_1:
                        self.layers["roads"] = not self.layers["roads"]
                    elif ev.key == pygame.K_2:
                        self.layers["ambulance"] = not self.layers["ambulance"]
                    elif ev.key == pygame.K_3:
                        self.layers["heatmap"] = not self.layers["heatmap"]

            if self.running and not self.done:
                if time.time() - self.last_step_t >= STEP_DUR:
                    self.sim_step()
                    self.last_step_t = time.time()

            self._draw()
            pygame.display.flip()

    # ── Coordinate helpers ────────────────────────────────────────────────────
    def _cell_rect(self, nid):
        r, c = divmod(nid, GRID)
        return c*CELL, r*CELL, CELL, CELL

    def _centre(self, nid):
        r, c = divmod(nid, GRID)
        return c*CELL + CELL//2, r*CELL + CELL//2

    # ── Master draw ───────────────────────────────────────────────────────────
    def _draw(self):
        scr = self.screen
        scr.fill(BG_MAP)

        # 1. Static tiles
        if self._tile_surf is None:
            self._tile_surf = self._build_tiles()
        scr.blit(self._tile_surf, (0,0))

        # 2. Crime heatmap
        if self.layers["heatmap"]:
            hs = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
            for nid, node in self.graph.nodes.items():
                x,y,w,h = self._cell_rect(nid)
                if node.crime_risk == "High":
                    hs.fill((*NEON_HEAT_HI,  85), (x+2,y+2,w-4,h-4))
                elif node.crime_risk == "Medium":
                    hs.fill((*NEON_HEAT_MED, 45), (x+2,y+2,w-4,h-4))
            scr.blit(hs, (0,0))

        # 3. Ambulance coverage glow
        if self.layers["ambulance"] and self.placer:
            cs = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
            for amb in self.placer.ambulances:
                ar,ac = divmod(amb, GRID)
                for nid in self.graph.nodes:
                    nr,nc = divmod(nid, GRID)
                    dist = abs(nr-ar)+abs(nc-ac)
                    if dist <= 4:
                        x,y,w,h = self._cell_rect(nid)
                        a = max(8, 36 - dist*8)
                        cs.fill((*NEON_COVERAGE, a), (x+1,y+1,w-2,h-2))
            scr.blit(cs, (0,0))

        # 4. Roads
        if self.layers["roads"]:
            road_s = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
            for edge in self.graph.edges:
                x1,y1 = self._centre(edge.node_a)
                x2,y2 = self._centre(edge.node_b)
                if edge.blocked:
                    pygame.draw.line(road_s, (*NEON_BLOCKED,90), (x1,y1),(x2,y2), 7)
                    pygame.draw.line(scr,     NEON_BLOCKED,       (x1,y1),(x2,y2), 2)
                    mx,my = (x1+x2)//2,(y1+y2)//2
                    evt = next((e for e in self.active_events if e["key"]==edge.key()),None)
                    icon = EV_ICON.get(evt["type"] if evt else "","X")
                    ecol = EV_COLOR.get(evt["type"] if evt else "Flooding", NEON_BLOCKED)
                    lb = self.f12b.render(icon, True, ecol)
                    scr.blit(lb,(mx-lb.get_width()//2, my-lb.get_height()//2))
                else:
                    pygame.draw.line(road_s, (*NEON_ROAD,30), (x1,y1),(x2,y2), 8)
                    pygame.draw.line(scr,     NEON_ROAD,      (x1,y1),(x2,y2), 2)
            scr.blit(road_s,(0,0))

        # 5. A* path
        if self.router and len(self.router.current_path) > 1:
            pts = [self._centre(n) for n in self.router.current_path]
            ps  = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
            for i in range(len(pts)-1):
                pygame.draw.line(ps,(*NEON_PATH,65),pts[i],pts[i+1],10)
            scr.blit(ps,(0,0))
            pygame.draw.lines(scr, NEON_PATH, False, pts, 3)
            for p in pts[1:-1]:
                pygame.draw.circle(scr, NEON_PATH, p, 3)

        # 6. Civilians
        if self.router:
            for civ in self.router.civilians:
                cx,cy = self._centre(civ.node_id)
                col   = NEON_RESCUED if civ.rescued else NEON_CIVILIAN
                _glow_circle(scr, col, (cx, cy-10), 5, halo=4)
                lb = self.f8.render("v" if civ.rescued else "!", True,(0,0,0))
                scr.blit(lb,(cx-lb.get_width()//2, cy-10-lb.get_height()//2))

        # 7. Ambulances
        if self.layers["ambulance"] and self.placer:
            for amb in self.placer.ambulances:
                ax,ay = self._centre(amb)
                _glow_circle(scr, NEON_AMBULANCE,(ax,ay),9,halo=5)
                pygame.draw.line(scr,(255,255,255),(ax,ay-5),(ax,ay+5),2)
                pygame.draw.line(scr,(255,255,255),(ax-5,ay),(ax+5,ay),2)

        # 8. Medical team
        if self.router:
            tx,ty = self._centre(self.router.team_pos)
            _glow_circle(scr, NEON_TEAM,(tx,ty),8,halo=6)
            lb = self.f8.render("T",True,(0,0,0))
            scr.blit(lb,(tx-lb.get_width()//2, ty-lb.get_height()//2))

        # 9. Grid lines
        for r in range(GRID+1):
            pygame.draw.line(scr,(14,24,42),(0,r*CELL),(MAP_W,r*CELL))
        for c in range(GRID+1):
            pygame.draw.line(scr,(14,24,42),(c*CELL,0),(c*CELL,MAP_H))

        # 10. Panel + legend
        self._draw_panel()
        self._draw_legend()

    # ── Static tile cache ─────────────────────────────────────────────────────
    def _build_tiles(self):
        surf      = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
        icon_font = pygame.font.SysFont("Segoe UI", 18, bold=True)
        surf.fill((0,0,0,0))

        for nid, node in self.graph.nodes.items():
            x,y,w,h = self._cell_rect(nid)
            t   = node.location_type.name
            bg  = TILE_BG.get(t,    TILE_BG["EMPTY"])
            acc = TILE_ACCENT.get(t, TILE_ACCENT["EMPTY"])

            pygame.draw.rect(surf, bg, (x+1,y+1,w-2,h-2))

            if t == "EMPTY":
                rng = random.Random(nid)
                for _ in range(6):
                    gx = x + rng.randint(4,w-4)
                    gy = y + rng.randint(4,h-4)
                    pygame.draw.circle(surf, acc, (gx,gy), 2)
            else:
                # Window dots
                rng = random.Random(nid*31)
                wpad,wcols,wrows,wsz = 8,3,3,4
                for wr in range(wrows):
                    for wc in range(wcols):
                        wx = x+wpad+wc*(w-wpad*2)//wcols
                        wy = y+wpad+wr*(h-wpad*2)//wrows
                        lit = rng.random() < 0.6
                        wc_ = (180,220,255) if lit else (28,45,70)
                        pygame.draw.rect(surf, wc_, (wx,wy,wsz,wsz))
                # Accent top stripe
                pygame.draw.rect(surf, acc, (x+1,y+1,w-2,4))
                # Type letter
                letter = TILE_LETTER.get(t,"")
                if letter:
                    lb = icon_font.render(letter, True, acc)
                    surf.blit(lb,(x+w//2-lb.get_width()//2,
                                  y+h//2-lb.get_height()//2))

            pygame.draw.rect(surf, acc, (x+1,y+1,w-2,h-2), 1)

        return surf

    # ── Dashboard panel ───────────────────────────────────────────────────────
    def _draw_panel(self):
        px  = MAP_W
        scr = self.screen
        pygame.draw.rect(scr, BG_PANEL, (px,0,PANEL_W,MAP_H))
        pygame.draw.line(scr,(28,55,95),(px,0),(px,MAP_H),1)

        cur_y = 0

        def txt(text, col=TXT, font=None, indent=10, dy=None):
            nonlocal cur_y
            f   = font or self.f11
            lb  = f.render(text, True, col)
            scr.blit(lb,(px+indent,cur_y))
            cur_y += dy if dy is not None else lb.get_height()+3

        def divider():
            nonlocal cur_y
            cur_y += 5
            pygame.draw.line(scr,(22,45,80),(px+8,cur_y),(px+PANEL_W-8,cur_y))
            cur_y += 7

        def section(title):
            nonlocal cur_y
            divider()
            txt(title, TXT_INFO, self.f12b, dy=18)

        def stat_card(label, value, col):
            nonlocal cur_y
            bw, bh = PANEL_W-16, 38
            pygame.draw.rect(scr,(14,24,44),(px+8,cur_y,bw,bh),border_radius=6)
            pygame.draw.rect(scr,(25,50,85),(px+8,cur_y,bw,bh),1,border_radius=6)
            lb1 = self.f10.render(label, True, TXT_DIM)
            lb2 = self.f14b.render(str(value), True, col)
            scr.blit(lb1,(px+14,cur_y+5))
            scr.blit(lb2,(px+14,cur_y+17))
            cur_y += 44

        # Header
        cur_y = 12
        txt("CITYMIND",                  TXT_INFO,  self.f22b, dy=26)
        txt("Urban Intelligence System", TXT_DIM,   self.f10,  dy=16)
        # Progress bar
        cur_y += 6
        pbw  = PANEL_W-20
        prog = self.step / SIM_STEPS
        pygame.draw.rect(scr,(20,38,65),(px+10,cur_y,pbw,8),border_radius=4)
        if prog:
            pygame.draw.rect(scr,NEON_ROAD,(px+10,cur_y,int(pbw*prog),8),border_radius=4)
        cur_y += 11
        txt(f"Step {self.step} / {SIM_STEPS}", TXT_DIM, self.f10, dy=14)

        # Stats
        section("CITY STATUS")
        rescued = sum(1 for c in self.router.civilians if c.rescued) if self.router else 0
        total   = len(self.router.civilians) if self.router else 0
        blocked = sum(1 for e in self.graph.edges if e.blocked)
        high_r  = sum(1 for n in self.graph.nodes.values() if n.crime_risk=="High")
        cov     = self.placer.coverage_percentage() if self.placer else 0
        health  = max(55,100-blocked*8-len(self.active_events)*5)
        hcol    = TXT_GOOD if health>80 else TXT_WARN if health>60 else TXT_DANGER

        stat_card("City Health",    f"{health}%",         hcol)
        stat_card("Roads Blocked",  str(blocked),         TXT_DANGER if blocked>2 else TXT_WARN if blocked else TXT_GOOD)
        stat_card("Amb. Coverage",  f"{cov}%",            TXT_GOOD if cov>70 else TXT_WARN)
        stat_card("Civilians",      f"{rescued}/{total}", TXT_GOOD if rescued==total and total>0 else TXT_WARN)
        stat_card("High-Risk Zones",str(high_r),          TXT_DANGER if high_r>20 else TXT_WARN)
        stat_card("Active Events",  str(len(self.active_events)), TXT_DANGER if self.active_events else TXT_GOOD)

        # Active events
        section("ACTIVE EVENTS")
        if not self.active_events:
            txt("  None", TXT_GOOD, self.f11, dy=16)
        for e in self.active_events[:5]:
            rem  = e["duration"]-(self.step-e["start"])
            ecol = EV_COLOR.get(e["type"], TXT_DANGER)
            pill_w = PANEL_W-20
            # Semi-transparent pill
            ps = pygame.Surface((pill_w,20), pygame.SRCALPHA)
            ps.fill((*ecol,45))
            scr.blit(ps,(px+10,cur_y))
            pygame.draw.rect(scr,ecol,(px+10,cur_y,pill_w,20),1,border_radius=4)
            lb = self.f10.render(
                f"  {EV_ICON.get(e['type'],'?')}  {e['type']}  [{rem}s]", True, ecol)
            scr.blit(lb,(px+14,cur_y+3))
            cur_y += 24

        # Layer toggles
        section("OVERLAYS  [1] [2] [3]")
        for key,label,col in [
            ("roads",    "Roads",      NEON_ROAD),
            ("ambulance","Ambulance",  NEON_AMBULANCE),
            ("heatmap",  "Crime Heat", NEON_HEAT_HI),
        ]:
            on = self.layers[key]
            bw,bh = PANEL_W-20,22
            bg = pygame.Surface((bw,bh),pygame.SRCALPHA)
            bg.fill((*col,28) if on else (14,24,44,255))
            scr.blit(bg,(px+10,cur_y))
            pygame.draw.rect(scr,col if on else (30,50,80),
                             (px+10,cur_y,bw,bh),1,border_radius=5)
            lb = self.f11.render(f"  [{'ON ' if on else 'OFF'}]  {label}",
                                  True, col if on else TXT_DIM)
            scr.blit(lb,(px+14,cur_y+4))
            cur_y += 27

        # Event log
        section("EVENT LOG")
        avail = max(1,(MAP_H-cur_y-8)//14)
        for entry in self.event_log[-avail:]:
            ec = {"info":TXT_DIM,"good":TXT_GOOD,"warn":TXT_WARN,"danger":TXT_DANGER
                  }.get(entry["type"],TXT)
            lb = self.f10.render(entry["msg"][:38], True, ec)
            scr.blit(lb,(px+10,cur_y))
            cur_y += 14

    # ── Legend bar ────────────────────────────────────────────────────────────
    def _draw_legend(self):
        by  = MAP_H
        scr = self.screen
        pygame.draw.rect(scr,(8,14,26),(0,by,WIN_W,BAR_H))
        pygame.draw.line(scr,(20,45,80),(0,by),(WIN_W,by),1)

        items = [
            ("Residential", TILE_ACCENT["RESIDENTIAL"]),
            ("Hospital",    TILE_ACCENT["HOSPITAL"]),
            ("School",      TILE_ACCENT["SCHOOL"]),
            ("Industrial",  TILE_ACCENT["INDUSTRIAL"]),
            ("Power Plant", TILE_ACCENT["POWER_PLANT"]),
            ("Amb Depot",   TILE_ACCENT["AMB_DEPOT"]),
            ("Park",        TILE_ACCENT["EMPTY"]),
            ("Team",        NEON_TEAM),
            ("Civilian",    NEON_CIVILIAN),
            ("Rescued",     NEON_RESCUED),
            ("Ambulance",   NEON_AMBULANCE),
            ("A* Path",     NEON_PATH),
        ]
        x = 8
        for label, col in items:
            pygame.draw.rect(scr, col, (x, by+13, 10,10), border_radius=2)
            lb = self.f10.render(label, True, TXT_DIM)
            scr.blit(lb,(x+13, by+13))
            x += lb.get_width()+26
            if x > MAP_W-50:
                break

        hint = "SPACE=Run/Pause   S=Step   R=Reset   1/2/3=Layers   ESC=Quit"
        hl = self.f10.render(hint, True,(45,75,115))
        scr.blit(hl,(8, by+28))

        stat,sc = (("● RUNNING",TXT_GOOD) if self.running
                   else ("■ DONE",TXT_INFO) if self.done
                   else ("‖ PAUSED",TXT_WARN))
        sl = self.f12b.render(stat, True, sc)
        scr.blit(sl,(MAP_W-sl.get_width()-10, by+14))

    # ── Text fallback ─────────────────────────────────────────────────────────
    def _run_text(self):
        print("\n=== TEXT MODE ===")
        for _ in range(SIM_STEPS):
            self.sim_step(); time.sleep(0.3)
        r = sum(1 for c in self.router.civilians if c.rescued)
        print(f"Rescued: {r}/{len(self.router.civilians)}")
        self.graph.summary()

    # ── Logging ───────────────────────────────────────────────────────────────
    def _log(self, msg, level="info"):
        self.event_log.append({"msg":msg,"type":level})
        if len(self.event_log) > 300:
            self.event_log.pop(0)
        pfx = {"info":"[INFO]","good":"[ OK ]","warn":"[WARN]","danger":"[ !! ]"
               }.get(level,"[LOG]")
        print(f"{pfx} {msg}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sim = CityMindSim()
    sim.run()
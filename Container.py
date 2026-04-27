import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from collections import Counter, defaultdict
import re, os, threading
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                    Paragraph, Spacer)
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    REPORTLAB_INSTALLED = True
except ImportError:
    REPORTLAB_INSTALLED = False

# ── Container Specs (cm & kg) ─────────────────────────────────────────────────
CONTAINERS = {
    "20GP": {"L": 589.8,  "W": 233.0, "H": 235.0, "MaxWt": 28000},
    "40GP": {"L": 1203.2, "W": 233.0, "H": 235.0, "MaxWt": 28800},
    "40HC": {"L": 1203.2, "W": 233.0, "H": 265.0, "MaxWt": 28600},
}
APP_VERSION = "v1.1.0"

# ── Weight tier thresholds (kg) ───────────────────────────────────────────────
TIER_HEAVY  = 600
TIER_MEDIUM = 350

SKU_COLORS = [
    "#4A9EFF","#3ECF8E","#F5A623","#F26B6B","#A78BFA",
    "#34D399","#FB923C","#60A5FA","#E879F9","#38BDF8",
    "#FBBF24","#6EE7B7","#FCA5A5","#C4B5FD","#93C5FD",
    "#FCD34D","#6B7280","#10B981","#EF4444","#8B5CF6",
]

# =============================================================================
#  DATA MODEL
# =============================================================================

class Item:
    def __init__(self, sku, l, w, h, weight=0.0, qty=1):
        self.sku          = sku
        self.dims         = [float(l), float(w), float(h)]
        self.vol          = float(l) * float(w) * float(h)
        self.weight       = float(weight)
        self.qty          = int(qty)
        self.pos          = [0.0, 0.0, 0.0]
        self.curr_dims    = [float(l), float(w), float(h)]
        self.tier         = self._compute_tier()
        self.current_load = 0.0
        self.max_stack_load = max(float(weight), 300.0) if weight > 0 else 300.0

    def _compute_tier(self):
        if   self.weight >= TIER_HEAVY:  return 3
        elif self.weight >= TIER_MEDIUM: return 2
        else:                            return 1

    def clone(self):
        return Item(self.sku, self.dims[0], self.dims[1], self.dims[2], self.weight)

    @property
    def tier_label(self):
        return {3: "Heavy", 2: "Medium", 1: "Light"}.get(self.tier, "?")


# =============================================================================
#  OPTIMISATION ENGINE  — Real-World Physics
#  KEY RULE: A pallet may only be stacked on another pallet if their
#  L×W footprints match within tolerance (same-size stacking only).
#  No bridging. No partial overhangs. Think like a forklift operator.
# =============================================================================

class OptimizationEngine:
    """
    Real-world column-slice packing engine.

    Mental model: imagine loading a container like a human forklift operator:
      1. Start at the BACK wall (x=0).
      2. Build complete CROSS-SECTION COLUMNS (fill Y width, then stack up)
         before advancing X toward the door.
      3. Same-footprint (L x W) pallets stack directly on top of each other.
         Height does NOT matter for stacking eligibility — only L x W match.
      4. Odd/leftover pallets always go at the FRONT of the current load face
         (highest X already used), never floating in the middle.
      5. Heavy pallets on the floor. Lighter pallets may stack on matching ones.
      6. No overhangs. No bridging across different footprints.
    """

    DOOR_CLEARANCE = 5.0
    FOOTPRINT_TOL  = 6.0   # cm — two pallets are "same size" if L and W match within this

    def __init__(self, c_name, c_info):
        self.c_name  = c_name
        self.L       = c_info["L"] - self.DOOR_CLEARANCE
        self.W       = c_info["W"]
        self.H       = c_info["H"]
        self.max_wt  = c_info["MaxWt"]
        self.placed  : list[Item] = []
        self.unplaced: list[Item] = []
        self.current_weight = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Geometry helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _fp_match(self, l1, w1, l2, w2):
        """True if two pallets share the same L×W footprint (±FOOTPRINT_TOL).
        Height is intentionally excluded — same floor area = stackable."""
        t = self.FOOTPRINT_TOL
        return ((abs(l1-l2) <= t and abs(w1-w2) <= t) or
                (abs(l1-w2) <= t and abs(w1-l2) <= t))

    def _overlaps(self, x, y, z, l, w, h):
        """True if the box (x,y,z,l,w,h) overlaps any placed item or container walls."""
        EPS = 0.15
        if (x < -EPS or y < -EPS or z < -EPS or
                x+l > self.L+EPS or y+w > self.W+EPS or z+h > self.H+EPS):
            return True
        for p in self.placed:
            if (x+l > p.pos[0]+EPS and x < p.pos[0]+p.curr_dims[0]-EPS and
                y+w > p.pos[1]+EPS and y < p.pos[1]+p.curr_dims[1]-EPS and
                z+h > p.pos[2]+EPS and z < p.pos[2]+p.curr_dims[2]-EPS):
                return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Support check — strict real-world physics
    # ─────────────────────────────────────────────────────────────────────────

    def _support(self, x, y, z, l, w, item: Item):
        """
        Returns (ok, supporter_or_None).
        Floor → always OK.
        Above floor → must land exactly on ONE pallet with matching L×W footprint.
        The item's footprint must be fully contained within the supporter (no overhang).
        """
        if z < 0.5:
            return True, None

        t = self.FOOTPRINT_TOL
        for p in self.placed:
            top = p.pos[2] + p.curr_dims[2]
            if abs(top - z) > 0.5:
                continue
            pl, pw = p.curr_dims[0], p.curr_dims[1]
            if not self._fp_match(l, w, pl, pw):
                continue
            # Must be fully inside supporter footprint
            if (x >= p.pos[0] - t and x+l <= p.pos[0]+pl+t and
                    y >= p.pos[1] - t and y+w <= p.pos[1]+pw+t):
                if item.tier > p.tier:           # heavier can't go on lighter
                    continue
                if p.current_load + item.weight > p.max_stack_load:
                    continue
                return True, p
        return False, None

    # ─────────────────────────────────────────────────────────────────────────
    # CoG
    # ─────────────────────────────────────────────────────────────────────────

    def get_center_of_gravity(self):
        if not self.placed:
            return self.L/2, self.W/2, 0.0
        tw = sum(i.weight for i in self.placed) or 1.0
        cx = sum((i.pos[0]+i.curr_dims[0]/2)*i.weight for i in self.placed) / tw
        cy = sum((i.pos[1]+i.curr_dims[1]/2)*i.weight for i in self.placed) / tw
        cz = sum((i.pos[2]+i.curr_dims[2]/2)*i.weight for i in self.placed) / tw
        return cx, cy, cz

    # ─────────────────────────────────────────────────────────────────────────
    # COLUMN-SLICE SPACE MAP
    # ─────────────────────────────────────────────────────────────────────────

    def _build_space_map(self):
        """
        Returns a dict: (col_x, col_y) → next available Z in that floor column.
        col_x / col_y are the X/Y origins of placed pallets — we use placed
        pallet edges as the grid lines of our columns.
        This lets us know exactly which floor slots are free and how high they are.
        """
        # Collect all unique X and Y breakpoints from placed items
        xs = sorted({0.0} | {round(p.pos[0]+p.curr_dims[0], 1) for p in self.placed})
        ys = sorted({0.0} | {round(p.pos[1]+p.curr_dims[1], 1) for p in self.placed})
        return xs, ys

    # ─────────────────────────────────────────────────────────────────────────
    # Candidate positions — derived from actual placed-item edges ONLY
    # No random mid-air points ever generated.
    # ─────────────────────────────────────────────────────────────────────────

    def _candidates(self, l, w, h):
        """
        Generate ALL physically meaningful placement positions:
          Floor positions: at X/Y edges of existing pallets (or origin).
          Stack positions: directly on top of a matching-footprint pallet.
        """
        pts = []

        # ── Floor candidates ──────────────────────────────────────────────
        # Collect every X-edge and every Y-edge from placed items
        x_edges = sorted({0.0} | {round(p.pos[0]+p.curr_dims[0], 1) for p in self.placed})
        y_edges = sorted({0.0} | {round(p.pos[1]+p.curr_dims[1], 1) for p in self.placed}
                         | {round(p.pos[1], 1) for p in self.placed})

        # Add right-wall snap
        yw = round(self.W - w, 1)
        if yw > 0:
            y_edges.append(yw)

        for x in x_edges:
            for y in sorted(set(y_edges)):
                if x+l <= self.L+0.1 and y >= -0.1 and y+w <= self.W+0.1:
                    pts.append((x, max(0.0, y), 0.0))

        # ── Stack candidates — ONLY on matching footprint pallets ─────────
        for p in self.placed:
            px, py = p.pos[0], p.pos[1]
            pl, pw, ph = p.curr_dims
            top_z = round(p.pos[2] + ph, 1)
            if self._fp_match(l, w, pl, pw) and top_z+h <= self.H+0.1:
                pts.append((px, py, top_z))

        # Deduplicate and validate
        seen = set()
        valid = []
        for (x, y, z) in pts:
            x = round(max(0.0, x), 1)
            y = round(max(0.0, y), 1)
            z = round(max(0.0, z), 1)
            key = (x, y, z)
            if key in seen:
                continue
            seen.add(key)
            if x+l <= self.L+0.1 and y+w <= self.W+0.1 and z+h <= self.H+0.1:
                valid.append([x, y, z])
        return valid

    # ─────────────────────────────────────────────────────────────────────────
    # PACK — column-first scoring
    # ─────────────────────────────────────────────────────────────────────────

    def pack(self, items: list, use_block_loading=True):
        if not items:
            return

        # ── Sort items: heavy first, then by SKU group for block loading ──
        if use_block_loading:
            groups = defaultdict(list)
            for it in items:
                groups[it.sku].append(it)
            # Score each group: (max tier, total volume) — heaviest+biggest first
            grp_score = {s: (max(i.tier for i in g), sum(i.vol for i in g))
                         for s, g in groups.items()}
            order = sorted(groups, key=lambda k: grp_score[k], reverse=True)
            sorted_items = []
            for s in order:
                sorted_items.extend(sorted(groups[s],
                                           key=lambda i: (i.tier, i.vol), reverse=True))
        else:
            sorted_items = sorted(items, key=lambda i: (i.tier, i.vol), reverse=True)

        target_cg_x = self.L / 2.0

        for item in sorted_items:
            if self.current_weight + item.weight > self.max_wt:
                self.unplaced.append(item)
                continue

            best_pos   = None
            best_dims  = None
            best_sup   = None
            best_score = float("inf")

            # Both XY orientations, height never rotated
            for (l, w, h) in [(item.dims[0], item.dims[1], item.dims[2]),
                               (item.dims[1], item.dims[0], item.dims[2])]:

                for (x, y, z) in self._candidates(l, w, h):
                    if self._overlaps(x, y, z, l, w, h):
                        continue

                    tmp = Item(item.sku, l, w, h, item.weight)
                    ok, supporter = self._support(x, y, z, l, w, tmp)
                    if not ok:
                        continue

                    # ── Column-first scoring ──────────────────────────────
                    #
                    # PRINCIPLE: Fill the current cross-section (Y×Z) completely
                    # before advancing X. This is how a real human loads a truck.
                    #
                    # Score components (lower = better placement):
                    #
                    # 1. X position — strongly prefer low X (fill from back)
                    #    BUT: a position is only "low X" if the column at that X
                    #    is actually being filled, not creating an island.
                    #
                    # 2. Current X frontier — what is the minimum X that has
                    #    unfilled space in the current cross-section?
                    #    Place there first.
                    #
                    # 3. Stack bonus — always prefer to stack on an existing
                    #    matching pallet before opening a new floor slot.
                    #
                    # 4. Wall hug — prefer y=0 or y=W-w over mid-aisle.
                    #
                    # 5. Adjacency — must be touching an existing pallet in X or Y.
                    #    Lone islands get a huge penalty.

                    # Find the "current frontier X" = minimum X where there is
                    # still room in the cross-section (not fully occupied in Y)
                    if self.placed:
                        floor_items = [p for p in self.placed if p.pos[2] < 0.5]
                        if floor_items:
                            # For each unique X-slice, measure how much Y is occupied
                            # The frontier is the smallest X-slice that is not full
                            x_slices = defaultdict(float)
                            for p in floor_items:
                                x_slices[round(p.pos[0], 1)] += p.curr_dims[1]
                            frontier_x = min(
                                (xi for xi, yw_used in x_slices.items()
                                 if yw_used < self.W - 10),  # 10cm slack
                                default=max(x_slices.keys(), default=0.0)
                            )
                        else:
                            frontier_x = 0.0
                    else:
                        frontier_x = 0.0

                    # Primary: how far beyond the frontier are we?
                    # Items exactly at frontier_x get 0; further = worse
                    x_beyond = max(0.0, x - frontier_x)
                    score = x_beyond * 2000.0

                    # Secondary: prefer low absolute X (fill from back)
                    score += x * 10.0

                    # Stack bonus — very strong: always fill vertically first
                    if z > 0.5 and supporter is not None:
                        score -= 50000.0

                    # Wall-hug: penalise mid-aisle positions
                    wall_dist = min(y, max(0.0, self.W - w - y))
                    score += wall_dist * 15.0

                    # Adjacency: a floor-level pallet must touch an existing one
                    # (in X or Y), otherwise it's a floating island → big penalty
                    if z < 0.5 and self.placed:
                        adjacent = False
                        for p in self.placed:
                            if p.pos[2] > 0.5:
                                continue
                            px0, py0 = p.pos[0], p.pos[1]
                            px1 = px0 + p.curr_dims[0]
                            py1 = py0 + p.curr_dims[1]
                            # X-adjacent (touching face to face) with Y overlap
                            y_ov = (y < py1 - 0.5 and y+w > py0 + 0.5)
                            x_touch = (abs(x - px1) < 1.0 or abs(x+l - px0) < 1.0)
                            # Y-adjacent (side by side) with X overlap
                            x_ov = (x < px1 - 0.5 and x+l > px0 + 0.5)
                            y_touch = (abs(y - py1) < 1.0 or abs(y+w - py0) < 1.0)
                            if (x_touch and y_ov) or (y_touch and x_ov):
                                adjacent = True
                                break
                        if not adjacent:
                            # Only allow non-adjacent if it's a fresh start
                            # at x=0 (very first pallet or new row at back)
                            if x > 1.0:
                                score += 100000.0  # virtually banned

                    # Same-SKU block cohesion (mild)
                    if use_block_loading:
                        same = [p for p in self.placed if p.sku == item.sku]
                        if same:
                            nearest = min(
                                abs(x-p.pos[0]) + abs(y-p.pos[1]) + abs(z-p.pos[2])
                                for p in same)
                            score += nearest * 0.1

                    # Axle balance (mild)
                    if item.weight > 150 and self.current_weight > 0:
                        cg_x, _, _ = self.get_center_of_gravity()
                        new_cg = ((cg_x * self.current_weight +
                                   (x + l/2) * item.weight) /
                                  (self.current_weight + item.weight))
                        score += abs(new_cg - target_cg_x) * 1.0

                    if score < best_score:
                        best_score = score
                        best_pos   = [x, y, z]
                        best_dims  = [l, w, h]
                        best_sup   = supporter

            if best_pos is not None:
                item.pos       = best_pos
                item.curr_dims = best_dims
                self.placed.append(item)
                self.current_weight += item.weight
                if best_sup is not None:
                    best_sup.current_load += item.weight
            else:
                self.unplaced.append(item)


# =============================================================================
#  HOVER TOOLTIP
# =============================================================================

class PalletTooltip:
    def __init__(self, fig, ax, placed_items, c_map, tk_widget):
        self.fig    = fig
        self.ax     = ax
        self.items  = placed_items
        self.c_map  = c_map
        self.widget = tk_widget
        self._tip   = None
        self._last  = -1
        self.selected_index = None
        fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        fig.canvas.mpl_connect('axes_leave_event',    lambda e: self._hide())
        fig.canvas.mpl_connect('button_press_event',  self._on_click)

    def _get_screen_info(self, item: Item):
            x0, y0, z0 = item.pos
            l, w, h = item.curr_dims

            pts2d = []
            depths = []

            corners = [(x0+dx, y0+dy, z0+dz)
                    for dx in (0, l) for dy in (0, w) for dz in (0, h)]

            proj = self.ax.get_proj()

            for (px, py, pz) in corners:
                try:
                    xd, yd, zd = proj3d.proj_transform(px, py, pz, proj)
                    sx, sy = self.ax.transData.transform((xd, yd))

                    if np.isfinite(sx) and np.isfinite(sy):
                        pts2d.append((sx, sy))
                        depths.append(zd)
                except:
                    pass

            if len(pts2d) < 2:
                return None, None

            xs = [p[0] for p in pts2d]
            ys = [p[1] for p in pts2d]

            bbox = (min(xs), min(ys), max(xs), max(ys))

            # 🔥 IMPORTANT: smaller depth = closer in matplotlib
            depth = min(depths) if depths else 0.0

            return bbox, depth
    def _hit_item(self, ex, ey):
        """
        Find the FRONTMOST pallet (highest projected depth = closest to viewer)
        whose screen bounding box contains the cursor.
        
        We use projected depth of the pallet center, not bbox area.
        This correctly handles overlapping pallets in 3D — the one visually
        on top (closest to the viewer camera) wins.
        """
        best_depth = float("inf")
        best_idx   = -1
        
        for i, item in enumerate(self.items):
            bbox, depth = self._get_screen_info(item)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            # Expand hit area slightly for easier clicking
            pad = 4
            if x0 - pad <= ex <= x1 + pad and y0 - pad <= ey <= y1 + pad:
                # Higher depth value = closer to the viewer camera
                if depth < best_depth:
                    best_depth = depth
                    best_idx   = i

        return best_idx

    def _on_move(self, event):
        if event.inaxes != self.ax:
            return

        idx = self._hit_item(event.x, event.y)

        if idx != -1:
            self.selected_index = idx
            self._last = -1
            self._show(self.items[idx], event)

            # 🔥 trigger redraw with highlight
            self._redraw_with_selection()
    def _redraw_with_selection(self):
        self.ax.cla()

        # Re-draw container box (reused logic)
        for i, item in enumerate(self.items):
            x, y, z = item.pos
            l, w, h = item.curr_dims

            base_color = self.c_map.get(item.sku, "#888")

            if i == self.selected_index:
                # 🔥 SELECTED STYLE
                self.ax.bar3d(
                    x, y, z, l, w, h,
                    color=base_color,
                    edgecolor="yellow",
                    linewidth=1.5,
                    alpha=1.0
                )
            else:
                self.ax.bar3d(
                    x, y, z, l, w, h,
                    color=base_color + "AA",
                    edgecolor="#222",
                    linewidth=0.3,
                    alpha=0.85
                )

        self.fig.canvas.draw_idle()
    def _on_click(self, event):
        if event.inaxes != self.ax: return
        idx = self._hit_item(event.x, event.y)
        if idx != -1:
            self._last = -1   # force refresh so same pallet re-shows on click
            self._show(self.items[idx], event)

    def _show(self, item: Item, event):
        self._hide()
        try:
            fig_h_px = self.fig.get_size_inches()[1] * self.fig.dpi
            rx = self.widget.winfo_rootx() + int(event.x) + 18
            ry = self.widget.winfo_rooty() + int(fig_h_px - event.y) + 18
        except Exception:
            return

        t = tk.Toplevel(self.widget)
        t.overrideredirect(True)
        t.wm_geometry(f"+{rx}+{ry}")
        t.configure(bg="#1E2128", highlightbackground="#4A9EFF", highlightthickness=1)

        load_pct = item.current_load / item.max_stack_load * 100 if item.max_stack_load else 0
        cap_left = max(0.0, item.max_stack_load - item.current_load)
        stk_txt  = "Yes" if item.pos[2] > 0.5 else "No"
        tier_color = {"Heavy": "#F26B6B", "Medium": "#F5A623", "Light": "#3ECF8E"}.get(
                      item.tier_label, "#E8EAF0")
        cap_color  = "#3ECF8E" if load_pct < 50 else ("#F5A623" if load_pct < 85 else "#F26B6B")
        sku_color  = self.c_map.get(item.sku, "#4A9EFF")

        rows = [
            ("Part Number",    item.sku,                                      "#FFFFFF"),
            ("Dimensions",     f"{item.curr_dims[0]:.0f} × "
                               f"{item.curr_dims[1]:.0f} × "
                               f"{item.curr_dims[2]:.0f} cm",                "#E8EAF0"),
            ("Weight",         f"{item.weight:.0f} kg",                       "#E8EAF0"),
            ("Tier",           f"{item.tier_label} ({item.weight:.0f} kg)",   tier_color),
            ("Position X/Y/Z", f"{item.pos[0]:.0f} / {item.pos[1]:.0f} / {item.pos[2]:.0f} cm", "#E8EAF0"),
            ("Stacked",        stk_txt,                                        "#F5A623" if stk_txt == "Yes" else "#9BA3B2"),
            ("Load on top",    f"{item.current_load:.0f} kg",                 "#E8EAF0"),
            ("Capacity left",  f"{cap_left:.0f} kg  ({100 - load_pct:.0f}% free)", cap_color),
        ]

        inner = tk.Frame(t, bg="#1E2128", padx=10, pady=8)
        inner.pack()
        tk.Frame(inner, bg=sku_color, width=3).grid(
            row=0, column=0, rowspan=len(rows)+1, sticky="ns", padx=(0, 8))
        tk.Label(inner, text=f"  {item.sku}", bg="#1E2128", fg="#FFFFFF",
                 font=("Segoe UI", 9, "bold")).grid(
            row=0, column=1, columnspan=2, sticky="w", pady=(0, 5))
        for r, (lbl, val, fg) in enumerate(rows[1:], start=1):
            tk.Label(inner, text=lbl + ":", bg="#1E2128", fg="#9BA3B2",
                     font=("Segoe UI", 8), width=14, anchor="w").grid(
                row=r, column=1, sticky="w")
            tk.Label(inner, text=val, bg="#1E2128", fg=fg,
                     font=("Segoe UI", 8, "bold"), anchor="w").grid(
                row=r, column=2, sticky="w", padx=(4, 0))

        self._tip = t

    def _hide(self):
        if self._tip:
            try: self._tip.destroy()
            except Exception: pass
            self._tip  = None
            self._last = -1


# =============================================================================
#  MAIN APPLICATION
# =============================================================================

class LogiPackApp:
    C_BG    = "#1E2128"; C_BG2   = "#252830"; C_BG3   = "#2D3139"
    C_BG4   = "#363B44"; C_BORDER= "#3A3F4A"
    C_TEXT  = "#F0F2F8"; C_TEXT2 = "#C8CEDD"; C_TEXT3 = "#8A94A8"
    C_BLUE  = "#5AAFFF"; C_GREEN = "#3ECF8E"
    C_AMBER = "#F5A623"; C_RED   = "#F26B6B"; C_SEL = "#1A5FA8"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LogiPack Pro — Container Loading Optimizer")
        self.root.geometry("1480x860")
        self.root.minsize(1200, 720)
        self.root.configure(bg=self.C_BG)
        self.items_data:      list[Item] = []
        self.containers_used: list[dict] = []
        self._tooltips:       list       = []
        self._setup_styles()
        self._build_ui()

    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        bg, bg2, bg3 = self.C_BG, self.C_BG2, self.C_BG3
        txt, txt2    = self.C_TEXT, self.C_TEXT2
        blue, border = self.C_BLUE, self.C_BORDER

        s.configure("Dark.TFrame",  background=bg)
        s.configure("Dark2.TFrame", background=bg2)
        s.configure("Dark3.TFrame", background=bg3)
        s.configure("TNotebook",    background=bg2, borderwidth=0)
        s.configure("TNotebook.Tab", background=bg2, foreground=txt2,
                    padding=[12, 6], font=("Segoe UI", 9))
        s.map("TNotebook.Tab",
              background=[("selected", bg)],
              foreground=[("selected", blue)])

        s.configure("Fleet.Treeview",
                    background=bg2, fieldbackground=bg2,
                    foreground=txt, rowheight=22, font=("Segoe UI", 9), borderwidth=0)
        s.configure("Fleet.Treeview.Heading",
                    background=bg3, foreground=txt2,
                    font=("Segoe UI", 8, "bold"), relief="flat")
        s.map("Fleet.Treeview",
              background=[("selected", self.C_SEL)],
              foreground=[("selected", "#fff")])

        s.configure("Dark.Vertical.TScrollbar",
                    background=bg3, troughcolor=bg2, arrowcolor=txt2, borderwidth=0)
        s.configure("Dark.TCombobox",
                    background=bg3, foreground=txt, fieldbackground=bg3,
                    arrowcolor=txt2, selectbackground=self.C_SEL, font=("Segoe UI", 9))
        s.configure("Dark.TCheckbutton",
                    background="#161920", foreground=txt2, font=("Segoe UI", 9))
        s.map("Dark.TCheckbutton",
              background=[("active", "#161920")],
              foreground=[("active", txt)])
        s.configure("Green.Horizontal.TProgressbar",
                    background=self.C_GREEN, troughcolor=bg3, borderwidth=0, thickness=8)

    def _build_ui(self):
        self._build_titlebar()
        self._build_toolbar()
        body = ttk.Frame(self.root, style="Dark.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        self._build_fleet_panel(body)
        self._build_center_panel(body)
        self._build_detail_panel(body)
        self._build_statusbar()

    def _build_titlebar(self):
        tb = tk.Frame(self.root, bg="#161920", height=38)
        tb.pack(fill=tk.X)
        tb.pack_propagate(False)
        tk.Label(tb, text="  LogiPack Pro", bg="#161920", fg=self.C_TEXT,
                 font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(tb, text=" — Container Loading Optimizer",
                 bg="#161920", fg=self.C_TEXT2, font=("Segoe UI", 9)).pack(side=tk.LEFT)

    def _build_toolbar(self):
        tb = tk.Frame(self.root, bg="#1a1d21", height=40)
        tb.pack(fill=tk.X)
        tb.pack_propagate(False)

        def tbtn(text, cmd, is_action=False, state=tk.NORMAL):
            b = tk.Button(tb, text=text,
                          bg="#1a1d21" if not is_action else self.C_SEL,
                          fg=self.C_TEXT2 if not is_action else "#fff",
                          activebackground=self.C_BG3, activeforeground=self.C_TEXT,
                          font=("Segoe UI", 9), relief=tk.FLAT, bd=0,
                          padx=10, pady=8,
                          cursor="hand2" if state == tk.NORMAL else "arrow",
                          state=state, command=cmd)
            b.pack(side=tk.LEFT)
            if state == tk.NORMAL:
                b.bind("<Enter>", lambda e: b.config(bg=self.C_BG3))
                b.bind("<Leave>", lambda e: b.config(
                    bg="#1a1d21" if not is_action else self.C_SEL))
            return b

        def sep():
            tk.Frame(tb, bg=self.C_BORDER, width=1).pack(
                side=tk.LEFT, fill=tk.Y, padx=4, pady=6)

        tbtn("Open Excel", self._upload)
        sep()
        self._btn_opt  = tbtn("Optimize",    self._run_multi, is_action=True)
        self._btn_stop = tbtn("Stop",         self._stop,     state=tk.DISABLED)
        sep()
        tk.Label(tb, text=" Manual:", bg="#1a1d21",
                 fg=self.C_TEXT2, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.c_var = tk.StringVar(value="40GP")
        ttk.Combobox(tb, textvariable=self.c_var,
                     values=list(CONTAINERS.keys()), width=6,
                     state="readonly", style="Dark.TCombobox").pack(side=tk.LEFT, padx=3)
        tbtn("Pack Single", self._run_single)
        sep()
        self.block_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tb, text="Block Loading",
                        variable=self.block_var,
                        style="Dark.TCheckbutton").pack(side=tk.LEFT, padx=6)
        sep()
        tbtn("Export Excel", self._export_excel)
        tbtn("Export PDF",   self._export_pdf,
             state=tk.NORMAL if REPORTLAB_INSTALLED else tk.DISABLED)
        tbtn("Save", self._save)

    def _build_fleet_panel(self, parent):
        frame = tk.Frame(parent, bg=self.C_BG2, width=230)
        frame.pack(side=tk.LEFT, fill=tk.Y)
        frame.pack_propagate(False)

        hdr = tk.Frame(frame, bg=self.C_BG3, height=32)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Fleet View", bg=self.C_BG3, fg=self.C_TEXT,
                 font=("Segoe UI", 9, "bold"), padx=10).pack(side=tk.LEFT, pady=6)

        cols = ("#", "Container", "Size", "Loaded", "Util%")
        self.fleet_tree = ttk.Treeview(frame, columns=cols, show="headings",
                                       style="Fleet.Treeview", selectmode="browse")
        for col, w in zip(cols, [28, 88, 46, 50, 62]):
            self.fleet_tree.heading(col, text=col)
            self.fleet_tree.column(col, width=w, anchor=tk.CENTER, stretch=False)
        self.fleet_tree.tag_configure("hi",  foreground=self.C_GREEN)
        self.fleet_tree.tag_configure("mid", foreground=self.C_AMBER)
        self.fleet_tree.tag_configure("lo",  foreground=self.C_RED)
        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL,
                           command=self.fleet_tree.yview,
                           style="Dark.Vertical.TScrollbar")
        self.fleet_tree.configure(yscrollcommand=sb.set)
        self.fleet_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        self.fleet_tree.bind("<<TreeviewSelect>>", self._on_sel)

        foot = tk.Frame(frame, bg=self.C_BG3, height=52)
        foot.pack(fill=tk.X, side=tk.BOTTOM)
        foot.pack_propagate(False)
        self._fct = tk.Label(foot, text="Total: —", bg=self.C_BG3,
                             fg=self.C_TEXT2, font=("Segoe UI", 8))
        self._fct.pack(anchor=tk.W, padx=10, pady=(6, 2))
        row = tk.Frame(foot, bg=self.C_BG3)
        row.pack(fill=tk.X, padx=10)
        tk.Label(row, text="Overall Util:", bg=self.C_BG3,
                 fg=self.C_TEXT2, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._fut = tk.Label(row, text="—", bg=self.C_BG3,
                             fg=self.C_GREEN, font=("Segoe UI", 9, "bold"))
        self._fut.pack(side=tk.RIGHT)

    def _build_center_panel(self, parent):
        frame = tk.Frame(parent, bg=self.C_TEXT)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        cards = tk.Frame(frame, bg=self.C_BG2, height=80)
        cards.pack(fill=tk.X)
        cards.pack_propagate(False)
        self._sc = {}
        for lbl, key, col in [
            ("Total Containers", "ct",      "Blue"),
            ("Total Pallets",    "pallets", ""),
            ("Total Weight kg",  "weight",  ""),
            ("Avg Utilization",  "util",    "Green"),
            ("Stacked Pallets",  "stacked", "Amber"),
        ]:
            self._sc[key] = self._card(cards, "—", lbl, col)

        self.nb = ttk.Notebook(frame)
        self.nb.pack(fill=tk.BOTH, expand=True)
        self.tab_iso = ttk.Frame(self.nb, style="Dark.TFrame")
        self.tab_top = ttk.Frame(self.nb, style="Dark.TFrame")
        self.nb.add(self.tab_iso, text="  Isometric View  ")
        self.nb.add(self.tab_top, text="  Top View  ")

    def _card(self, parent, val, lbl, color=""):
        c = tk.Frame(parent, bg=self.C_BG3, padx=14, pady=8)
        c.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=4, pady=6)
        fg = {"Blue": self.C_BLUE, "Green": self.C_GREEN, "Amber": self.C_AMBER}.get(
            color, self.C_TEXT)
        v = tk.Label(c, text=val, bg=self.C_BG3, fg=fg, font=("Segoe UI", 18, "bold"))
        v.pack(anchor=tk.W)
        tk.Label(c, text=lbl, bg=self.C_BG3, fg=self.C_TEXT2,
                 font=("Segoe UI", 8)).pack(anchor=tk.W)
        return v

    def _build_detail_panel(self, parent):
        frame = tk.Frame(parent, bg=self.C_BG2, width=260)
        frame.pack(side=tk.RIGHT, fill=tk.Y)
        frame.pack_propagate(False)

        hdr = tk.Frame(frame, bg=self.C_BG3, height=28)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Container Details", bg=self.C_BG3, fg=self.C_TEXT,
                 font=("Segoe UI", 9, "bold"), padx=10).pack(side=tk.LEFT, pady=4)

        tbl = tk.Frame(frame, bg=self.C_BG2)
        tbl.pack(fill=tk.X, padx=8, pady=6)
        self._dr: dict[str, tk.Label] = {}
        for key in ["Container ID", "Container Size", "Dimensions (cm)",
                    "Max Capacity", "Loaded Weight",
                    "Volume Util %", "Weight Util %",
                    "Pallets Loaded", "Stacked", "Remaining"]:
            row = tk.Frame(tbl, bg=self.C_BG2)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=key, bg=self.C_BG2, fg=self.C_TEXT2,
                     font=("Segoe UI", 8), width=18, anchor=tk.W).pack(side=tk.LEFT)
            v = tk.Label(row, text="—", bg=self.C_BG2, fg=self.C_TEXT,
                         font=("Segoe UI", 8, "bold"), anchor=tk.E)
            v.pack(side=tk.RIGHT)
            self._dr[key] = v
        
        tk.Frame(frame, bg=self.C_BORDER, height=1).pack(fill=tk.X, padx=8)
        
        br = tk.Frame(bg=self.C_BG2)
        br.pack(fill=tk.X, pady=3)
        self._bal = tk.Canvas(br, bg=self.C_BG4, height=10, highlightthickness=0)
        self._bal.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self._ballbl = tk.Label(br, text="—", bg=self.C_BG2,
                                fg=self.C_BLUE, font=("Segoe UI", 8, "bold"), width=5)
        self._ballbl.pack(side=tk.LEFT)

        pl_cols = ("Part Number", "L", "W", "H", "Wt", "Z", "Tier")
        self.pallet_tree = ttk.Treeview(frame, columns=pl_cols, show="headings",
                                        style="Fleet.Treeview", height=9)
        for col, w in zip(pl_cols, [68, 28, 28, 28, 42, 28, 46]):
            self.pallet_tree.heading(col, text=col)
            self.pallet_tree.column(col, width=w, anchor=tk.CENTER, stretch=False)
        self.pallet_tree.tag_configure("stacked", foreground=self.C_AMBER)
        self.pallet_tree.tag_configure("heavy",   foreground=self.C_RED)
        self.pallet_tree.tag_configure("medium",  foreground=self.C_AMBER)
        self.pallet_tree.tag_configure("light",   foreground=self.C_GREEN)
        sb2 = ttk.Scrollbar(frame, orient=tk.VERTICAL,
                            command=self.pallet_tree.yview,
                            style="Dark.Vertical.TScrollbar")
        self.pallet_tree.configure(yscrollcommand=sb2.set)
        pw = tk.Frame(frame, bg=self.C_BG2)
        pw.pack(fill=tk.BOTH, expand=True)
        self.pallet_tree.pack(in_=pw, side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb2.pack(in_=pw, side=tk.LEFT, fill=tk.Y)

        tk.Frame(frame, bg=self.C_BORDER, height=1).pack(fill=tk.X)
        lh = tk.Frame(frame, bg=self.C_BG3, height=26)
        lh.pack(fill=tk.X)
        tk.Label(lh, text="Logs", bg=self.C_BG3, fg=self.C_TEXT,
                 font=("Segoe UI", 8, "bold"), padx=10).pack(side=tk.LEFT, pady=4)
        self._logtxt = tk.Text(frame, bg=self.C_BG2, fg=self.C_TEXT2,
                               font=("Consolas", 8), height=6,
                               state=tk.DISABLED, bd=0, highlightthickness=0, wrap=tk.WORD)
        self._logtxt.tag_configure("ok",  foreground=self.C_GREEN)
        self._logtxt.tag_configure("err", foreground=self.C_RED)
        self._logtxt.tag_configure("ts",  foreground=self.C_TEXT3)
        self._logtxt.pack(fill=tk.X, padx=4, pady=4)
        bar = tk.Frame(frame, bg=self.C_BG2, height=48)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)
        bi = tk.Frame(bar, bg=self.C_BG2)
        bi.pack(fill=tk.X, padx=12)
        self._pv = tk.DoubleVar()
        
        self._pbl = tk.Label(bi, text="0%", bg=self.C_BG2, fg=self.C_GREEN,
                             font=("Segoe UI", 9, "bold"), width=5)
        self._pbl.pack(side=tk.LEFT, padx=(6, 0))
        tk.Frame(frame, bg=self.C_BORDER, height=1).pack(fill=tk.X)
        ph = tk.Frame(frame, bg=self.C_BG3, height=10)
        ph.pack(fill=tk.X)
        self._plhdr = tk.Label(ph, text="Pallet List", bg=self.C_BG3,
                               fg=self.C_TEXT, font=("Segoe UI", 8, "bold"), padx=5)
        self._plhdr.pack(side=tk.LEFT, pady=4)
        
    def _build_statusbar(self):
        sb = tk.Frame(self.root, bg="#161920", height=24)
        sb.pack(fill=tk.X, side=tk.BOTTOM)
        sb.pack_propagate(False)
        tk.Frame(sb, bg=self.C_GREEN, width=6, height=6).pack(
            side=tk.LEFT, padx=(10, 4), pady=9)
        self._stlbl = tk.Label(sb, text="Ready", bg="#161920",
                               fg=self.C_TEXT2, font=("Segoe UI", 8))
        self._stlbl.pack(side=tk.LEFT)
        tk.Label(sb, text=f"Creator - Kidzoro",
                 bg="#161920", fg=self.C_TEXT2, font=("Segoe UI", 8)).pack(
            side=tk.RIGHT, padx=14)
        self._tslbl = tk.Label(sb, text="", bg="#161920",
                               fg=self.C_TEXT2, font=("Segoe UI", 8))
        self._tslbl.pack(side=tk.RIGHT, padx=14)
        self._tick()

    def _tick(self):
        self._tslbl.config(text=datetime.now().strftime("%d-%b-%Y  %H:%M"))
        self.root.after(30000, self._tick)

    def _log(self, msg, level="ok"):
        ts = datetime.now().strftime("%H:%M:%S")
        self._logtxt.config(state=tk.NORMAL)
        self._logtxt.insert(tk.END, f"{ts}  ", "ts")
        self._logtxt.insert(tk.END, msg + "\n", level)
        self._logtxt.see(tk.END)
        self._logtxt.config(state=tk.DISABLED)
        self._stlbl.config(text=msg)

    def _upload(self):
        path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel", "*.xlsx *.xls"), ("All", "*.*")])
        if not path:
            return

        def sf(val):
            if pd.isna(val):
                return 0.0
            m = re.search(r"[-+]?\d*\.?\d+", str(val).replace(",", ""))
            return float(m.group()) if m else 0.0

        try:
            df = pd.read_excel(path)
            self.items_data = []
            skipped = 0
            for idx, r in df.iterrows():
                raw = r.get("Part Number", r.get("SKU", r.get("PART", "")))
                sku = f"Item_{idx+1}" if pd.isna(raw) or str(raw).strip() == "" else str(raw).strip()
                l   = sf(r.get("Length", r.get("L", 0)))
                w   = sf(r.get("Width",  r.get("W", 0)))
                h   = sf(r.get("Height", r.get("H", 0)))
                wt  = sf(r.get("Weight", r.get("Gross Wt / Plt", r.get("Wt", 0))))
                qty = int(sf(r.get("Qty", r.get("Quantity", 1))) or 1)
                if l > 0 and w > 0 and h > 0:
                    for _ in range(qty):
                        self.items_data.append(Item(sku, l, w, h, wt))
                else:
                    skipped += 1
            self._log(f"Loaded {len(self.items_data)} pallets"
                      + (f" ({skipped} skipped)" if skipped else ""))
            self._upd_cards()
            messagebox.showinfo("Loaded",
                                f"✔  {len(self.items_data)} pallets loaded.\nClick Optimize.")
        except Exception as e:
            self._log(f"Error: {e}", "err")
            messagebox.showerror("Error", str(e))

    def _pack_eng(self, c_name, items):
        eng = OptimizationEngine(c_name, CONTAINERS[c_name])
        eng.pack(items, use_block_loading=self.block_var.get())
        return eng

    def _run_single(self):
        if not self.items_data:
            messagebox.showwarning("No Data", "Load Excel first.")
            return
        self._clear()
        c = self.c_var.get()
        eng = self._pack_eng(c, [i.clone() for i in self.items_data])
        st = sum(1 for i in eng.placed if i.pos[2] > 0.5)
        self.containers_used.append({"id": 1, "c_name": c, "packer": eng, "c_info": CONTAINERS[c]})
        self._pop_fleet()
        self._upd_cards()
        self._log(f"Done — {len(eng.placed)} placed, {st} stacked, {len(eng.unplaced)} left.")

    def _run_multi(self):
        if not self.items_data:
            messagebox.showwarning("No Data", "Load Excel first.")
            return
        self._clear()
        self._log("Optimization started…")
        self._btn_opt.config(state=tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL)
        self._stop_flag = False
        threading.Thread(target=self._multi_thread, daemon=True).start()

    def _stop(self):
        self._stop_flag = True
        self._log("Stopped by user.", "err")

    def _multi_thread(self):
        remaining = [i.clone() for i in self.items_data]
        c_num = 1
        while remaining and not getattr(self, "_stop_flag", False):
            best_p = best_n = None
            for cn in ["20GP", "40GP", "40HC"]:
                p = self._pack_eng(cn, [i.clone() for i in remaining])
                if not p.unplaced:
                    best_p, best_n = p, cn
                    break
            if not best_p:
                best_n = "40HC"
                best_p = self._pack_eng(best_n, [i.clone() for i in remaining])
            if not best_p.placed:
                self.root.after(0, lambda: messagebox.showerror("Error", "Item exceeds 40HC."))
                break
            st  = sum(1 for i in best_p.placed if i.pos[2] > 0.5)
            msg = f"Container {c_num} ({best_n}): {len(best_p.placed)} pallets, {st} stacked"
            self.root.after(0, lambda m=msg: self._log(m))
            self.containers_used.append({"id": c_num, "c_name": best_n,
                                         "packer": best_p, "c_info": CONTAINERS[best_n]})
            remaining = best_p.unplaced
            c_num += 1
            self.root.after(0, lambda: (self._pop_fleet(), self._upd_cards()))
        self.root.after(0, self._done)

    def _done(self):
        self._btn_opt.config(state=tk.NORMAL)
        self._btn_stop.config(state=tk.DISABLED)
        n  = len(self.containers_used)
        ts = sum(sum(1 for i in d["packer"].placed if i.pos[2] > 0.5)
                 for d in self.containers_used)
        self._log(f"Complete — {n} container(s), {ts} pallets stacked vertically.")
        self._upd_cards()
        ch = self.fleet_tree.get_children()
        if ch:
            self.fleet_tree.selection_set(ch[0])
            self._on_sel(None)
        messagebox.showinfo("Done", f"✔  {n} containers used.\n{ts} pallets stacked.")

    def _clear(self):
        [self.fleet_tree.delete(c) for c in self.fleet_tree.get_children()]
        self.containers_used = []
        self._tooltips = []
        for tab in (self.tab_iso, self.tab_top):
            [w.destroy() for w in tab.winfo_children()]
        self._logtxt.config(state=tk.NORMAL)
        self._logtxt.delete("1.0", tk.END)
        self._logtxt.config(state=tk.DISABLED)

    def _pop_fleet(self):
        [self.fleet_tree.delete(c) for c in self.fleet_tree.get_children()]
        tp = tc = 0
        for d in self.containers_used:
            p = d["packer"]
            c = d["c_info"]
            pv = sum(i.vol for i in p.placed)
            cv = c["L"] * c["W"] * c["H"]
            u  = round(pv / cv * 100, 1) if cv else 0
            tag = "hi" if u >= 90 else ("mid" if u >= 80 else "lo")
            self.fleet_tree.insert("", tk.END, iid=str(d["id"] - 1),
                                   values=(d["id"], f"Container {d['id']}", d["c_name"],
                                           len(p.placed), f"{u}%"), tags=(tag,))
            tp += pv
            tc += cv
        self._fct.config(text=f"Total: {len(self.containers_used)} containers")
        if tc:
            ov = round(tp / tc * 100, 1)
            self._fut.config(text=f"{ov}%")
            self._pv.set(ov)
            self._pbl.config(text=f"{ov}%")

    def _upd_cards(self):
        if not self.containers_used:
            for k in self._sc:
                self._sc[k].config(text="—")
            self._sc["pallets"].config(text=str(len(self.items_data)))
            return
        n  = len(self.containers_used)
        tp = sum(len(d["packer"].placed)        for d in self.containers_used)
        tw = sum(d["packer"].current_weight     for d in self.containers_used)
        ts = sum(sum(1 for i in d["packer"].placed if i.pos[2] > 0.5)
                 for d in self.containers_used)
        vols = [sum(i.vol for i in d["packer"].placed) /
                (d["c_info"]["L"] * d["c_info"]["W"] * d["c_info"]["H"]) * 100
                for d in self.containers_used]
        avg_u = round(sum(vols) / len(vols), 1) if vols else 0
        self._sc["ct"].config(text=str(n))
        self._sc["pallets"].config(text=f"{tp:,}")
        self._sc["weight"].config(text=f"{tw:,.0f}")
        self._sc["util"].config(text=f"{avg_u}%")
        self._sc["stacked"].config(text=f"{ts:,}")
        self._pv.set(avg_u)
        self._pbl.config(text=f"{avg_u}%")

    def _on_sel(self, _):
        sel = self.fleet_tree.selection()
        if not sel:
            return
        d = self.containers_used[int(sel[0])]
        self._draw_views(d)
        self._fill_detail(d)
        self._fill_plist(d)

    def _draw_views(self, data):
        self._tooltips = []
        for tab in (self.tab_iso, self.tab_top):
            [w.destroy() for w in tab.winfo_children()]
        lbl = f"Container {data['id']} ({data['c_name']})"
        self._draw_one(self.tab_iso, data, 25, -60, f"{lbl} — Isometric", tooltip=True)
        self._draw_one(self.tab_top, data, 90, -90, f"{lbl} — Top View",  tooltip=False)

    def _draw_one(self, tab, data, elev, azim, title, tooltip=True):
        packer = data["packer"]
        c      = data["c_info"]
        bg     = self.C_BG

        fig = plt.Figure(figsize=(9, 5.2), facecolor=bg)
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(bg)
        fig.patch.set_facecolor(bg)

        L, W, H = c["L"], c["W"], c["H"]
        for xs, ys, zs in [
            ([0,L],[0,0],[0,0]), ([0,L],[W,W],[0,0]),
            ([0,L],[0,0],[H,H]), ([0,L],[W,W],[H,H]),
            ([0,0],[0,W],[0,0]), ([L,L],[0,W],[0,0]),
            ([0,0],[0,W],[H,H]), ([L,L],[0,W],[H,H]),
            ([0,0],[0,0],[0,H]), ([L,L],[0,0],[0,H]),
            ([0,0],[W,W],[0,H]), ([L,L],[W,W],[0,H]),
        ]:
            ax.plot3D(xs, ys, zs, color=self.C_BLUE, alpha=0.25, linewidth=0.6)

        skus  = sorted({i.sku for i in packer.placed})
        c_map = {s: SKU_COLORS[n % len(SKU_COLORS)] for n, s in enumerate(skus)}

        for item in sorted(packer.placed,
                           key=lambda i: -(i.pos[0] + i.pos[1] + i.pos[2])):
            x, y, z     = item.pos
            dl, dw, dh  = item.curr_dims
            col  = c_map.get(item.sku, "#888")
            edge = "#FFFFFF" if z > 0.5 else "#000000"
            lw   = 0.8       if z > 0.5 else 0.3
            ax.bar3d(x, y, z, dl, dw, dh,
                     color=col + "CC", edgecolor=edge,
                     alpha=0.92, linewidth=lw)

        ax.set_xlim(0, L); ax.set_ylim(0, W); ax.set_zlim(0, H)
        ax.set_box_aspect((L, W, H))
        ax.view_init(elev=elev, azim=azim)
        st = sum(1 for i in packer.placed if i.pos[2] > 0.5)
        ax.set_title(f"{title}  |  {st} stacked",
                     color=self.C_TEXT2, fontsize=9, pad=8)

        for spine in ax.spines.values():
            spine.set_visible(False)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor(self.C_BORDER)
        ax.tick_params(colors=self.C_TEXT2, labelsize=7)

        patches = [mpatches.Patch(color=c_map[s], label=s)
                   for s in list(c_map)[:8]]
        if patches:
            ax.legend(handles=patches, loc="upper left", fontsize=7, ncol=2,
                      facecolor=self.C_BG3, edgecolor=self.C_BORDER,
                      labelcolor=self.C_TEXT)

        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().configure(bg=bg)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

        if tooltip and packer.placed:
            tip = PalletTooltip(fig, ax, packer.placed, c_map,
                                canvas.get_tk_widget())
            self._tooltips.append(tip)

    def _fill_detail(self, data):
        p = data["packer"]
        c = data["c_info"]
        pv = sum(i.vol for i in p.placed)
        cv = c["L"] * c["W"] * c["H"]
        vu = pv / cv * 100 if cv else 0
        wu = p.current_weight / c["MaxWt"] * 100 if c["MaxWt"] else 0
        st = sum(1 for i in p.placed if i.pos[2] > 0.5)
        rm = len(p.unplaced)
        vals = {
            "Container ID":   f"Container {data['id']}",
            "Container Size":  data["c_name"],
            "Dimensions (cm)": f"{c['L']:.0f}×{c['W']:.0f}×{c['H']:.0f}",
            "Max Capacity":    f"{c['MaxWt']:,} kg",
            "Loaded Weight":   f"{p.current_weight:,.0f} kg",
            "Volume Util %":   f"{vu:.1f}%",
            "Weight Util %":   f"{wu:.1f}%",
            "Pallets Loaded":  str(len(p.placed)),
            "Stacked":         str(st),
            "Remaining":       str(rm),
        }
        for k, v in vals.items():
            lbl = self._dr[k]
            lbl.config(text=v, fg=self.C_TEXT)
            if k in ("Volume Util %", "Weight Util %"):
                p2 = float(v.replace("%", ""))
                lbl.config(fg=self.C_GREEN if p2 >= 90 else
                           (self.C_AMBER if p2 >= 70 else self.C_RED))
            elif k == "Remaining":
                lbl.config(fg=self.C_GREEN if rm == 0 else self.C_RED)
            elif k == "Stacked":
                lbl.config(fg=self.C_AMBER if st > 0 else self.C_TEXT3)

        cg_x, _, _ = p.get_center_of_gravity()
        pct = cg_x / c["L"] * 100 if c["L"] else 50
        self._ballbl.config(text=f"{pct:.0f}%",
                            fg=self.C_GREEN if 40 <= pct <= 60 else self.C_RED)
        self._bal.update_idletasks()
        bw = self._bal.winfo_width() or 120
        self._bal.delete("all")
        self._bal.create_rectangle(0, 0, int(bw * pct / 100), 10,
                                   fill=self.C_BLUE, outline="")
        self._bal.create_line(bw // 2, 0, bw // 2, 10, fill="white", width=1)
        self._plhdr.config(text=f"Pallet List — Container {data['id']}")

    def _fill_plist(self, data):
        [self.pallet_tree.delete(r) for r in self.pallet_tree.get_children()]
        for item in data["packer"].placed:
            tags = []
            if item.pos[2] > 0.5:
                tags.append("stacked")
            tags.append({3: "heavy", 2: "medium", 1: "light"}.get(item.tier, "light"))
            self.pallet_tree.insert("", tk.END,
                values=(item.sku,
                        f"{item.curr_dims[0]:.0f}", f"{item.curr_dims[1]:.0f}",
                        f"{item.curr_dims[2]:.0f}", f"{item.weight:.0f}",
                        f"{item.pos[2]:.0f}", item.tier_label),
                tags=tuple(tags))

    def _export_excel(self):
        if not self.containers_used:
            messagebox.showwarning("Nothing", "Run optimization first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                            filetypes=[("Excel", "*.xlsx")])
        if not path:
            return
        rows = []
        for d in self.containers_used:
            cg_x, cg_y, cg_z = d["packer"].get_center_of_gravity()
            for item in d["packer"].placed:
                rows.append({
                    "Container_ID": d["id"], "Type": d["c_name"],
                    "SKU": item.sku, "Weight_kg": item.weight,
                    "Tier": item.tier_label,
                    "Length_cm": item.curr_dims[0], "Width_cm": item.curr_dims[1],
                    "Height_cm": item.curr_dims[2],
                    "Pos_X": round(item.pos[0], 1), "Pos_Y": round(item.pos[1], 1),
                    "Pos_Z": round(item.pos[2], 1),
                    "Stacked": "Yes" if item.pos[2] > 0.5 else "No",
                    "Load_on_top_kg": round(item.current_load, 1),
                    "Capacity_remaining_kg": round(max(0, item.max_stack_load - item.current_load), 1),
                    "CoG_X": round(cg_x, 1), "CoG_Y": round(cg_y, 1), "CoG_Z": round(cg_z, 1),
                })
        pd.DataFrame(rows).to_excel(path, index=False)
        self._log(f"Exported: {os.path.basename(path)}")
        messagebox.showinfo("Done", f"Saved:\n{os.path.basename(path)}")

    def _export_pdf(self):
        if not REPORTLAB_INSTALLED:
            messagebox.showerror("Missing", "pip install reportlab")
            return
        if not self.containers_used:
            messagebox.showwarning("Nothing", "Run optimization first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                            filetypes=[("PDF", "*.pdf")])
        if not path:
            return
        doc = SimpleDocTemplate(path, pagesize=A4)
        st  = getSampleStyleSheet()
        els = []
        els.append(Paragraph("LogiPack Pro — Load Manifest", st["Title"]))
        els.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"), st["Normal"]))
        els.append(Spacer(1, 14))
        for d in self.containers_used:
            p = d["packer"]
            c = d["c_info"]
            cg_x, _, _ = p.get_center_of_gravity()
            axle = round(cg_x / c["L"] * 100, 1) if c["L"] else 50
            stk  = sum(1 for i in p.placed if i.pos[2] > 0.5)
            els.append(Paragraph(f"Container {d['id']} ({d['c_name']})", st["Heading2"]))
            els.append(Paragraph(
                f"Pallets: {len(p.placed)} ({stk} stacked) | "
                f"Weight: {p.current_weight:,.0f}/{c['MaxWt']:,} kg | Balance: {axle}%",
                st["Normal"]))
            els.append(Spacer(1, 6))
            rows = [["SKU", "Qty", "Total Wt", "Dims (cm)", "Tier", "Stacked"]]
            for sku, cnt in sorted(Counter(i.sku for i in p.placed).items()):
                s    = next(i for i in p.placed if i.sku == sku)
                stk_c = sum(1 for i in p.placed if i.sku == sku and i.pos[2] > 0.5)
                rows.append([sku, str(cnt), f"{s.weight * cnt:,.0f} kg",
                              f"{s.dims[0]:.0f}×{s.dims[1]:.0f}×{s.dims[2]:.0f}",
                              s.tier_label, f"{stk_c}/{cnt}"])
            t = Table(rows, colWidths=[3.5*cm, 1.2*cm, 2.5*cm, 3.2*cm, 2*cm, 2*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1A5FA8")),
                ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
                ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",   (0,0), (-1,-1), 8),
                ("ALIGN",      (0,0), (-1,-1), "CENTER"),
                ("ROWBACKGROUNDS", (0,1), (-1,-1),
                 [colors.white, colors.HexColor("#F4F6FA")]),
                ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#CCC")),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ]))
            els.append(t)
            els.append(Spacer(1, 20))
        doc.build(els)
        self._log(f"PDF: {os.path.basename(path)}")
        messagebox.showinfo("Done", f"PDF saved:\n{os.path.basename(path)}")

    def _save(self):
        if not self.items_data:
            messagebox.showwarning("Nothing", "Load data first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                            filetypes=[("Excel", "*.xlsx")])
        if not path:
            return
        pd.DataFrame([{"Part Number": i.sku, "Length": i.dims[0], "Width": i.dims[1],
                        "Height": i.dims[2], "Weight": i.weight}
                      for i in self.items_data]).to_excel(path, index=False)
        self._log(f"Saved: {os.path.basename(path)}")
        messagebox.showinfo("Saved", "Project saved.")

# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    LogiPackApp(root)
    root.mainloop()

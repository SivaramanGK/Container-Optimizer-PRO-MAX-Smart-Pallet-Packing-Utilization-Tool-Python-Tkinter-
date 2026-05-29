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
import re, os, io, threading, random
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                    Paragraph, Spacer, Image as RLImage)
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    REPORTLAB_INSTALLED = True
except ImportError:
    REPORTLAB_INSTALLED = False

# ── Container Specs ───────────────────────────────────────────────────────────
CONTAINERS = {
    "20GP": {"L": 589.8,  "W": 235.0, "H": 239.0, "MaxWt": 28000},
    "40GP": {"L": 1203.2, "W": 235.0, "H": 239.0, "MaxWt": 28800},
    "40HC": {"L": 1203.2, "W": 235.0, "H": 269.8, "MaxWt": 28600},
}

AUTO_CONTAINER_SIZES = ["20GP", "40HC", "40GP"]

# ── Weight Tier Thresholds ────────────────────────────────────────────────────
#   Heavy  ≥ 500 kg  → floor only, never stacked
#   Medium 200–499 kg
#   Light  < 200 kg  → top fill preferred
TIER_HEAVY  = 500
TIER_MEDIUM = 200

# ── Stacking Constants ────────────────────────────────────────────────────────
MAX_STACK_LAYERS = 4        # absolute depth limit
FOOTPRINT_TOL    = 1.5      # cm tolerance for footprint matching

# ── Utilisation Targets ───────────────────────────────────────────────────────
UTIL_TARGET_MIN = 85.0      # % — reallocate below this
UTIL_TARGET_MAX = 95.0      # % — stop trying above this

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
    """
    Physical stacking rule (user requirement):
        The total weight stacked ABOVE any box must not exceed that box's
        own weight.  This is enforced via  max_stack_load = item.weight.

    Example:  bottom=500 kg  →  max 500 kg cumulative load above it.
              bottom=500, middle=350, top=300 :
                load on bottom = 350+300 = 650 > 500  →  REJECTED ✗
              bottom=500, middle=350, top=100 :
                load on bottom = 350+100 = 450 ≤ 500  →  OK       ✓

    Same-weight-same-footprint rule (automatic via max_stack_load):
        bottom=520, middle=520 → load on bottom = 520 ≤ 520  OK (2 allowed)
        Adding any 3rd item :  load on bottom = 520 + X > 520  ALWAYS fails
    """
    def __init__(self, sku, l, w, h, weight=0.0, qty=1,
                 date="Any", dest="Any", shipment_ref="—"):
        self.sku          = sku
        self.dims         = [float(l), float(w), float(h)]
        self.vol          = float(l)*float(w)*float(h)
        self.weight       = float(weight)
        self.qty          = int(qty)
        self.pos          = [0.0, 0.0, 0.0]
        self.curr_dims    = [float(l), float(w), float(h)]
        self.tier         = self._compute_tier()
        self.current_load = 0.0
        self.date         = str(date)
        self.dest         = str(dest)
        self.shipment_ref = str(shipment_ref)

        # ── PHYSICAL STACKING RULE ────────────────────────────────────────────
        # A box cannot safely carry more cumulative weight than its own weight.
        self.max_stack_load = max(float(weight), 60.0)

        self.supporter      = None
        self._pre_depth     = 1
        self.is_door_single = False
        self._normalize_dims()

    def _compute_tier(self):
        if   self.weight >= TIER_HEAVY:  return 3
        elif self.weight >= TIER_MEDIUM: return 2
        else:                            return 1

    def clone(self):
        c = Item(self.sku, self.dims[0], self.dims[1], self.dims[2],
                 self.weight, qty=1, date=self.date, dest=self.dest,
                 shipment_ref=getattr(self, "shipment_ref", "—"))
        c._pre_depth = getattr(self, "_pre_depth", 1)
        return c

    def _normalize_dims(self):
        STANDARDS = [78, 80, 98, 100, 102, 114, 120]
        for i in [0, 1]:
            v    = self.dims[i]
            best = min(STANDARDS, key=lambda s: abs(s - v))
            if abs(best - v) <= 3:
                self.dims[i] = float(best)

    @property
    def tier_label(self):
        return {3:"Heavy", 2:"Medium", 1:"Light"}.get(self.tier, "?")

    @property
    def footprint_key(self):
        """Canonical (min, max) footprint for same-footprint detection."""
        return (min(self.dims[0], self.dims[1]), max(self.dims[0], self.dims[1]))


# =============================================================================
#  TOWER  (vertical stack of items at one floor position)
# =============================================================================
class Tower:
    """
    Stacking rules enforced here:
    1.  An item can only be added if its weight ≤ the top item's weight
        (lighter-on-top hierarchy; equal weight is allowed).
    2.  The physical load rule (max_stack_load) is checked in _build_towers.
    3.  Heavy items (tier 3) can never be stacked; they are always the base.
    4.  dest and date must match the tower's group.
    """

    def __init__(self):
        self.items        = []
        self.total_height = 0.0
        self.total_weight = 0.0
        self.length       = 0.0
        self.width        = 0.0
        self.dest         = None
        self.date         = None

    def can_accept(self, item):
        if not self.items:
            return True
        if item.dims[0] > self.length + 5: return False
        if item.dims[1] > self.width  + 5: return False
        if self.dest != item.dest:         return False
        if self.date != item.date:         return False
        base = self.items[-1]
        if item.tier   > base.tier:        return False   # tier must not go up
        if item.weight > base.weight:      return False   # must be lighter or equal
        return True

    def add(self, item):
        if not self.items:
            self.length = item.dims[0]
            self.width  = item.dims[1]
            self.dest   = item.dest
            self.date   = item.date
        else:
            self.length = max(self.length, item.dims[0])
            self.width  = max(self.width,  item.dims[1])
        self.items.append(item)
        self.total_height += item.dims[2]
        self.total_weight += item.weight


# =============================================================================
#  OPTIMISATION ENGINE
# =============================================================================
class OptimizationEngine:
    DOOR_CLEARANCE    = 0.0
    FP_TOL            = FOOTPRINT_TOL
    EPS               = 0.1
    GRID_X            = 114
    GRID_Y            = 98
    MIN_FLOOR_FILL    = 0.22
    MIN_SUPPORT_HEAVY  = 0.70
    MIN_SUPPORT_MEDIUM = 0.55
    MIN_SUPPORT_LIGHT  = 0.35
    MAX_COLUMN_WEIGHT  = {1: 3500, 2: 5000, 3: 9000}

    _W_DEPTH_ALIGN   = 16.0
    _W_DEAD_SPACE    = 0.40
    _W_FRAGMENTATION = 1.8
    _W_CG            = 1.0
    _W_SKYLINE       = 7.0
    _W_CAVITY        = 550.0
    _W_CEILING       = 0.4
    _MIN_USABLE_DIM  = 60.0
    _FUTURE_STD_L    = 114.0
    _FUTURE_STD_W    = 98.0

    def __init__(self, c_name, c_info):
        self.c_name        = c_name
        self.L             = c_info["L"] - self.DOOR_CLEARANCE
        self.W             = c_info["W"]
        self.H             = c_info["H"]
        self.max_wt        = c_info["MaxWt"]
        self.placed        : list[Item] = []
        self.unplaced      : list[Item] = []
        self.current_weight = 0.0
        self._max_placed_x  = 0.0
        self._singles_start_x = 0.0
        self.free_spaces   : list[dict] = []
        self.zone_weights  = {"HEAVY": 0, "MEDIUM": 0, "LIGHT": 0}

    # ── Zone helpers ──────────────────────────────────────────────────────────
    def _get_weight_zone(self, x):
        r = x / self.L
        if r <= 0.40:             return "HEAVY"
        elif r <= 0.80:           return "MEDIUM"
        return "LIGHT"

    def _target_weight_for_zone(self, zone):
        return {"HEAVY": 750.0, "MEDIUM": 320.0, "LIGHT": 90.0}.get(zone, 350.0)

    # ── Geometry ──────────────────────────────────────────────────────────────
    def _fp_match(self, tl, tw, bl, bw):
        t = self.FP_TOL
        return ((tl <= bl+t and tw <= bw+t) or (tl <= bw+t and tw <= bl+t))

    def _overlaps(self, x, y, z, l, w, h):
        E = self.EPS
        if x<-E or y<-E or z<-E: return True
        if x+l>self.L+E or y+w>self.W+E or z+h>self.H+E: return True
        for p in self.placed:
            if (x+l>p.pos[0]+E and x<p.pos[0]+p.curr_dims[0]-E and
                y+w>p.pos[1]+E and y<p.pos[1]+p.curr_dims[1]-E and
                z+h>p.pos[2]+E and z<p.pos[2]+p.curr_dims[2]-E):
                return True
        return False

    def _fits(self, x, y, z, l, w, h):
        return not self._overlaps(x, y, z, l, w, h)

    def _floor_fill_ratio(self):
        fa = self.L * self.W
        ua = sum(p.curr_dims[0]*p.curr_dims[1]
                 for p in self.placed if p.pos[2] < 0.5)
        return ua / fa if fa > 0 else 0.0

    def _support_ratio(self, x, y, l, w, base):
        bx, by = base.pos[0], base.pos[1]
        bl, bw = base.curr_dims[0], base.curr_dims[1]
        ox = max(0, min(x+l, bx+bl) - max(x, bx))
        oy = max(0, min(y+w, by+bw) - max(y, by))
        return (ox*oy/(l*w)) if l*w > 0 else 0

    def _column_weight_ok(self, supporter, item_weight):
        total = item_weight
        curr  = supporter
        while curr is not None:
            total += curr.weight + curr.current_load
            curr   = curr.supporter
        return total <= self.MAX_COLUMN_WEIGHT.get(supporter.tier, 1200)

    def _volume_fill_rate(self):
        cv = self.L*self.W*self.H
        pv = sum(i.curr_dims[0]*i.curr_dims[1]*i.curr_dims[2] for i in self.placed)
        return pv/cv if cv else 0.0

    # ── Support check ─────────────────────────────────────────────────────────
    def _get_support(self, x, y, z, l, w, item):
        """
        Physical stacking rules:
          • item must weigh ≤ its supporter (equal allowed → same-weight limited to 2
            automatically by the max_stack_load cumulative check)
          • cumulative load on every ancestor ≤ that ancestor's own weight
          • heavy items never leave the floor
        """
        if z < 0.5:
            return True, None

        for p in self.placed:
            top = p.pos[2] + p.curr_dims[2]
            if abs(top - z) > 0.5:
                continue
            ratio = self._support_ratio(x, y, l, w, p)
            min_req = (self.MIN_SUPPORT_HEAVY  if item.tier == 3 else
                       self.MIN_SUPPORT_MEDIUM if item.tier == 2 else
                       self.MIN_SUPPORT_LIGHT)
            if ratio < min_req:                          continue
            if item.tier > p.tier:                       continue   # tier hierarchy
            if item.weight > p.weight:                   continue   # must be ≤ base weight
            if item.tier == 3 and z > 0.5:               continue   # heavy never stacks
            if not self._column_weight_ok(p, item.weight): continue

            depth = 1; walker = p
            while walker.supporter is not None:
                depth += 1; walker = walker.supporter
            if depth >= MAX_STACK_LAYERS:
                continue

            # ── Physical load check ──────────────────────────────────────────
            # Each ancestor must be able to carry the additional weight.
            # max_stack_load = ancestor.weight  (own-weight rule)
            overload = False
            curr = p
            while curr is not None:
                if curr.current_load + item.weight > curr.max_stack_load:
                    overload = True; break
                curr = curr.supporter
            if overload:
                continue

            return True, p
        return False, None

    def _validate_placement(self, item, x, y, z, l, w, h, supporter):
        if x+l > self.L:  return False
        if y+w > self.W:  return False
        if z+h > self.H:  return False
        if self.current_weight + item.weight > self.max_wt: return False
        if item.tier == 3 and z > 0.5:   return False
        if supporter:
            if item.tier   > supporter.tier:   return False
            if item.weight > supporter.weight: return False
        return True

    # ── Candidate positions ───────────────────────────────────────────────────
    def _candidates(self, l, w, h):
        pts = [(0.0, 0.0, 0.0)]
        fi  = [p for p in self.placed if p.pos[2] < 0.5]
        if fi:
            x_edges = sorted({0.0} | {round(p.pos[0]+p.curr_dims[0],1) for p in fi})
            y_edges = sorted({0.0} |
                             {round(p.pos[1],1) for p in fi} |
                             {round(p.pos[1]+p.curr_dims[1],1) for p in fi} |
                             {round(max(0.0,self.W-w),1)})
            frontier = max(p.pos[0]+p.curr_dims[0] for p in fi)
            for xe in x_edges:
                if xe > frontier+0.5: break
                for ye in y_edges:
                    pts.append((xe,ye,0.0))
            for p in fi:
                yl = p.pos[1]-w
                if yl >= -self.EPS:
                    pts.append((p.pos[0], max(0.0,yl), 0.0))
                pts.extend([
                    (p.pos[0],               p.pos[1]+p.curr_dims[1], 0.0),
                    (p.pos[0]+p.curr_dims[0], p.pos[1],               0.0),
                    (p.pos[0]+p.curr_dims[0], p.pos[1]+p.curr_dims[1],0.0),
                ])

        for p in self.placed:
            top_z = round(p.pos[2]+p.curr_dims[2],1)
            if top_z+h > self.H+0.1: continue
            if self._fp_match(l,w,p.curr_dims[0],p.curr_dims[1]):
                pts.append((p.pos[0],p.pos[1],top_z))

        seen, valid = set(), []
        for (x,y,z) in pts:
            x = round(max(0.0,x),1); y = round(max(0.0,y),1); z = round(max(0.0,z),1)
            if z<0.5 and abs(l-114)<5 and abs(w-98)<5:
                x = round(round(x/self.GRID_X)*self.GRID_X,1)
                y = round(round(y/self.GRID_Y)*self.GRID_Y,1)
            k = (x,y,z)
            if k in seen: continue
            seen.add(k)
            if x+l<=self.L+0.1 and y+w<=self.W+0.1 and z+h<=self.H+0.1:
                valid.append((x,y,z))
        return valid

    # ── Scoring ───────────────────────────────────────────────────────────────
    def _skyline_variance(self, x, l, top_h):
        p = 0.0
        for pi in self.placed:
            if pi.pos[0]<x+l and pi.pos[0]+pi.curr_dims[0]>x:
                p += abs((pi.pos[2]+pi.curr_dims[2]) - top_h)
        return p

    def _score(self, item, x, y, z, l, w, h):
        rem_x = self.L-(x+l); rem_y = self.W-(y+w)
        dead       = rem_x*rem_y
        frag       = (500 if 1<rem_x<80 else 0) + (700 if 1<rem_y<60 else 0)
        cg_x,_,_   = self.get_center_of_gravity()
        cg_pen     = abs(cg_x - self.L/2)*18
        depth_pen  = abs(x - self._max_placed_x)
        wall_gap   = (200 if 0<x<40 else 0)+(200 if 0<y<30 else 0)

        stack_pen = 0
        if z > 0.5:
            if item.weight >= TIER_MEDIUM:
                mf = 0.45 if h<120 else 0.35
                if self._floor_fill_ratio() < mf: stack_pen += 2500
            else:
                if self._floor_fill_ratio() < 0.15: stack_pen += 600

        cav_pen = 0
        if 0<rem_x<self._MIN_USABLE_DIM: cav_pen += self._W_CAVITY
        if 0<rem_y<self._MIN_USABLE_DIM: cav_pen += self._W_CAVITY

        fut_bonus = 0
        if rem_x>=self._FUTURE_STD_L and rem_y>=self._FUTURE_STD_W: fut_bonus -= 300
        if x==0 and rem_x>=2*self._FUTURE_STD_L:                    fut_bonus -= 200

        ceil_pen = 0
        if z > 0.5:
            hr = self.H-(z+h)
            if hr > 90: ceil_pen = hr*self._W_CEILING

        stack_pot = -150 if (z<0.5 and (self.H-h)>80) else 0

        zone   = self._get_weight_zone(x)
        wt_pen = abs(item.weight - self._target_weight_for_zone(zone)) * 3.5

        # heavy-base bonus: reward stacking light items directly on heavy bases
        hb_bonus = 0
        if z > 0.5 and item.weight < TIER_MEDIUM:
            for p in self.placed:
                if abs(p.pos[2]+p.curr_dims[2]-z) < 0.5 and p.weight >= TIER_HEAVY:
                    hb_bonus -= 350; break

        clus = 0
        for p in self.placed:
            if abs(p.pos[0]-x) < 160:
                if abs(p.curr_dims[0]-l)<3 and abs(p.curr_dims[1]-w)<3: clus -= 350
                if abs(p.curr_dims[2]-h)<8:                              clus -= 200
                if abs(p.weight-item.weight)<80:                         clus -= 220

        smooth = sum(abs(p.weight-item.weight)
                     for p in self.placed if abs(p.pos[0]-x)<120)

        floor_frag = 0
        if z <= 0:
            for p in self.placed:
                gap = abs((p.pos[0]+p.curr_dims[0])-x)
                if 5<gap<80: floor_frag += 400

        return (
            wt_pen + smooth*9.0 +
            clus + hb_bonus +
            depth_pen*self._W_DEPTH_ALIGN +
            dead*self._W_DEAD_SPACE + frag*self._W_FRAGMENTATION +
            cg_pen*self._W_CG + wall_gap + stack_pen +
            self._skyline_variance(x,l,z+h)*self._W_SKYLINE +
            cav_pen + ceil_pen + stack_pot + fut_bonus + floor_frag
            - x*0.3 - abs(y)*0.1
        )

    # ── Free-space registry ───────────────────────────────────────────────────
    def _register_free_spaces(self, item):
        MAX = 220; MIN_D = 20.0
        x,y,z = item.pos; l,w,h = item.curr_dims
        cands = []
        if self.L-(x+l)>MIN_D: cands.append({"x":x+l,"y":y,"z":z,"l":self.L-(x+l),"w":w,"h":h})
        if self.W-(y+w)>MIN_D: cands.append({"x":x,"y":y+w,"z":z,"l":l,"w":self.W-(y+w),"h":h})
        if self.H-(z+h)>MIN_D: cands.append({"x":x,"y":y,"z":z+h,"l":l,"w":w,"h":self.H-(z+h)})
        for ns in cands:
            if any(es["x"]<=ns["x"] and es["y"]<=ns["y"] and es["z"]<=ns["z"] and
                   es["x"]+es["l"]>=ns["x"]+ns["l"] and
                   es["y"]+es["w"]>=ns["y"]+ns["w"] and
                   es["z"]+es["h"]>=ns["z"]+ns["h"] for es in self.free_spaces):
                continue
            self.free_spaces = [
                es for es in self.free_spaces
                if not (ns["x"]<=es["x"] and ns["y"]<=es["y"] and ns["z"]<=es["z"] and
                        ns["x"]+ns["l"]>=es["x"]+es["l"] and
                        ns["y"]+ns["w"]>=es["y"]+es["w"] and
                        ns["z"]+ns["h"]>=es["z"]+es["h"])
            ]
            self.free_spaces.append(ns)
        if len(self.free_spaces) > MAX:
            self.free_spaces.sort(key=lambda s: -(s["l"]*s["w"]*s["h"]))
            self.free_spaces = self.free_spaces[:int(MAX*0.75)]

    def _orientations(self, item):
        l0,w0 = item.dims[0],item.dims[1]
        rw = self.W % min(l0,w0)
        return [(l0,w0),(w0,l0)] if abs(rw-l0)<=abs(rw-w0) else [(w0,l0),(l0,w0)]

    # ── Place routine ─────────────────────────────────────────────────────────
    def _place(self, item, use_bl=False, cg_tgt=False, door_only=False,
               floor_only=False, stacking_only=False, prefer_heavy_base=False):
        if self.current_weight + item.weight > self.max_wt:
            self.unplaced.append(item); return

        # Fast-path: registered free spaces for small items
        if item.curr_dims[2] < 90:
            for gap in self.free_spaces:
                if (item.curr_dims[0]<=gap["l"] and
                    item.curr_dims[1]<=gap["w"] and
                    item.curr_dims[2]<=gap["h"]):
                    x,y,z = gap["x"],gap["y"],gap["z"]
                    if floor_only    and z > 0:  continue
                    if stacking_only and z <= 0: continue
                    if self._fits(x,y,z,*item.curr_dims):
                        ok,sup = self._get_support(x,y,z,item.curr_dims[0],item.curr_dims[1],item)
                        if ok:
                            item.pos=[x,y,z]; item.supporter=sup
                            self.placed.append(item); self._register_free_spaces(item)
                            self.current_weight += item.weight
                            c=sup
                            while c: c.current_load+=item.weight; c=c.supporter
                            self._max_placed_x=max(self._max_placed_x,x+item.curr_dims[0])
                            return

        best_pos=best_dims=best_sup=None; best_score=float('inf')
        for (l,w) in self._orientations(item):
            h = item.dims[2]
            for (x,y,z) in self._candidates(l,w,h):
                if door_only     and x < getattr(self,"_singles_start_x",0.0)-1.0: continue
                if floor_only    and z > 0:  continue
                if stacking_only and z <= 0: continue
                if self._overlaps(x,y,z,l,w,h): continue
                probe = Item(item.sku,l,w,h,item.weight)
                ok,sup = self._get_support(x,y,z,l,w,probe)
                if not self._validate_placement(item,x,y,z,l,w,h,sup): continue
                if not ok: continue
                if item.weight >= TIER_HEAVY and x > self.L*0.45: continue
                lp = 0
                if item.weight < TIER_MEDIUM and x < self.L*0.25: lp += 600
                if prefer_heavy_base and z>0.5:
                    if not any(abs(p.pos[2]+p.curr_dims[2]-z)<0.5 and p.weight>=TIER_HEAVY
                               for p in self.placed):
                        lp += 1800
                s = self._score(item,x,y,z,l,w,h) + lp
                if s < best_score:
                    best_score,best_pos,best_dims,best_sup=s,[x,y,z],[l,w,h],sup

        if best_pos is not None:
            item.pos=best_pos; item.curr_dims=best_dims
            item.supporter=best_sup; item.is_door_single=door_only
            self.placed.append(item); self._register_free_spaces(item)
            self.current_weight += item.weight
            self.zone_weights[self._get_weight_zone(best_pos[0])] += item.weight
            c=best_sup
            while c: c.current_load+=item.weight; c=c.supporter
            self._max_placed_x=max(self._max_placed_x,best_pos[0]+best_dims[0])
        else:
            self.unplaced.append(item)

    # ── Compaction ────────────────────────────────────────────────────────────
    def _compact_floor(self):
        improved=True; passes=0
        while improved and passes<25:
            improved=False; passes+=1
            for item in sorted([p for p in self.placed if p.pos[2]<0.5],key=lambda p:p.pos[0]):
                l,w,h=item.curr_dims; mx=0.0
                for o in self.placed:
                    if o is item: continue
                    yov=(item.pos[1]<o.pos[1]+o.curr_dims[1]-0.1 and item.pos[1]+w>o.pos[1]+0.1)
                    zov=(0.0<o.pos[2]+o.curr_dims[2]-0.1 and h>o.pos[2]+0.1)
                    if yov and zov and o.pos[0]+o.curr_dims[0]<=item.pos[0]+0.15:
                        mx=max(mx,o.pos[0]+o.curr_dims[0])
                if round(mx,1)<item.pos[0]-1.0: item.pos[0]=round(mx,1); improved=True
            for item in sorted([p for p in self.placed if p.pos[2]<0.5],key=lambda p:p.pos[1]):
                l,w,h=item.curr_dims; my=0.0
                for o in self.placed:
                    if o is item: continue
                    xov=(item.pos[0]<o.pos[0]+o.curr_dims[0]-0.1 and item.pos[0]+l>o.pos[0]+0.1)
                    zov=(0.0<o.pos[2]+o.curr_dims[2]-0.1 and h>o.pos[2]+0.1)
                    if xov and zov and o.pos[1]+o.curr_dims[1]<=item.pos[1]+0.15:
                        my=max(my,o.pos[1]+o.curr_dims[1])
                if round(my,1)<item.pos[1]-1.0: item.pos[1]=round(my,1); improved=True
            for item in sorted([p for p in self.placed if p.pos[2]>0.5],key=lambda p:p.pos[2]):
                if item.supporter:
                    s=item.supporter
                    if abs(item.pos[0]-s.pos[0])>0.5 or abs(item.pos[1]-s.pos[1])>0.5:
                        item.pos[0]=s.pos[0]; item.pos[1]=s.pos[1]; improved=True
        if self.placed:
            self._max_placed_x=max(p.pos[0]+p.curr_dims[0] for p in self.placed)

    # ── Tower builder ─────────────────────────────────────────────────────────
    def _build_towers(self, items):
        """
        Build towers respecting:
          1. Weight-decreasing order (lighter items on top)
          2. Equal weight allowed → max_stack_load limits depth automatically
          3. Physical load rule enforced by cumulative overload check
          4. Heavy items (tier 3) are always tower bases, never stacked
        """
        towers   = []
        doorstep = []

        sorted_items = sorted(items, key=lambda x: (
            x.dest, x.date, -x.tier,
            -(x.dims[0]*x.dims[1]), -x.weight, -x.dims[2],
        ))

        for item in sorted_items:
            best_tower = None; best_score = float("inf")

            for tower in towers:
                if not tower.can_accept(item): continue
                new_height = tower.total_height + item.dims[2]
                if new_height > self.H + self.EPS: continue
                if item.tier == 3 and tower.items: continue   # heavy → floor only
                if tower.items and item.weight > tower.items[-1].weight: continue

                # ── Physical load check ────────────────────────────────────
                overload = False; load_above = item.weight
                for ti in reversed(tower.items):
                    if load_above > ti.max_stack_load:
                        overload = True; break
                    load_above += ti.weight
                if overload: continue

                # ── Column weight limit ────────────────────────────────────
                base_tier = tower.items[0].tier if tower.items else item.tier
                if tower.total_weight + item.weight > self.MAX_COLUMN_WEIGHT.get(base_tier,1200):
                    continue

                # ── Scoring: prefer tall towers + small weight diff ────────
                wsm = abs(tower.items[-1].weight - item.weight) if tower.items else 0

                # Bonus: heavy base + light item = good 3-layer opportunity
                hb  = -300 if (tower.items and tower.items[0].weight>=TIER_HEAVY
                                and item.weight < TIER_MEDIUM) else 0
                # Bonus: 3rd layer (short item on tall+light)
                sl  = -400 if (tower.items and tower.items[-1].weight<TIER_MEDIUM
                                and tower.items[-1].dims[2]>=90
                                and item.dims[2]<=70) else 0

                sc = -new_height + wsm*0.004 + hb + sl
                if sc < best_score: best_score=sc; best_tower=tower

            if best_tower:
                best_tower.add(item)
            else:
                t=Tower(); t.add(item); towers.append(t)

        return towers, doorstep

    # ── MaxRects helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _mr_prune(rects):
        n=len(rects); keep=[True]*n
        for i in range(n):
            if not keep[i]: continue
            r1=rects[i]
            for j in range(n):
                if i==j or not keep[j]: continue
                r2=rects[j]
                if (r2["x"]<=r1["x"] and r2["y"]<=r1["y"] and
                    r2["x"]+r2["l"]>=r1["x"]+r1["l"] and
                    r2["y"]+r2["w"]>=r1["y"]+r1["w"]):
                    if r2["l"]>r1["l"] or r2["w"]>r1["w"]:
                        keep[i]=False; break
        return [rects[i] for i in range(n) if keep[i]]

    def _mr_update(self, rects, px, py, pl, pw):
        out=[]
        for fr in rects:
            fx,fy,fl,fw=fr["x"],fr["y"],fr["l"],fr["w"]
            if px>=fx+fl or px+pl<=fx or py>=fy+fw or py+pw<=fy:
                out.append(fr); continue
            if px>fx:     out.append({"x":fx,"y":fy,"l":px-fx,"w":fw})
            if px+pl<fx+fl: out.append({"x":px+pl,"y":fy,"l":fx+fl-(px+pl),"w":fw})
            if py>fy:     out.append({"x":fx,"y":fy,"l":fl,"w":py-fy})
            if py+pw<fy+fw: out.append({"x":fx,"y":py+pw,"l":fl,"w":fy+fw-(py+pw)})
        return self._mr_prune(out)

    def _mr_best_rect(self, rects, tl, tw):
        best=None; bs=float("inf")
        for fr in rects:
            for (cl,cw) in [(tl,tw),(tw,tl)]:
                if cl<=fr["l"]+self.EPS and cw<=fr["w"]+self.EPS:
                    sc=min(fr["l"]-cl,fr["w"]-cw)
                    if sc<bs: bs=sc; best=(fr["x"],fr["y"],cl,cw)
        return best

    # ── Block pack ────────────────────────────────────────────────────────────
    def _block_pack(self, items):
        """
        STAIRCASE PATTERN (matches the target image):
        Towers are sorted DESCENDING by total height.
        MaxRects places tallest towers first → naturally creates a staircase
        from the back wall to the door: Tower-A (tallest) at x≈0,
        Tower-B (next) at x≈114, … decreasing toward the door.

        This matches the visual in the uploaded image where the tallest
        stacks are at the rear and height decreases toward the camera.
        """
        # Pre-orient items for best floor density
        for item in items:
            l,w=item.dims[0],item.dims[1]
            l1,w1=max(l,w),min(l,w); l2,w2=min(l,w),max(l,w)
            c1=int(self.L/l1)*int(self.W/w1) if l1 and w1 else 0
            c2=int(self.L/l2)*int(self.W/w2) if l2 and w2 else 0
            if c2>c1:   item.dims[0],item.dims[1]=l2,w2
            elif c1>c2: item.dims[0],item.dims[1]=l1,w1
            else:
                if self.W%w2<self.W%w1: item.dims[0],item.dims[1]=l2,w2
            item.curr_dims=list(item.dims)

        towers, doorstep = self._build_towers(items)

        # ── STAIRCASE SORT: tallest towers placed first (back wall → door) ───
        towers.sort(key=lambda t: (
            -round(t.total_height),          # 1st: tallest first  → staircase
            -round(t.length), -round(t.width),
            -round(t.total_weight/100),
        ))

        free_rects=[{"x":0.0,"y":0.0,"l":self.L,"w":self.W}]
        placed_xs=[]

        for tower in towers:
            best = self._mr_best_rect(free_rects, tower.length, tower.width)
            if best is None:
                for it in tower.items: self.unplaced.append(it)
                continue
            bx,by,bl,bw = best
            if bx+bl > self.L+self.EPS:
                for it in tower.items: self.unplaced.append(it)
                continue

            z=0.0; supporter=None
            for item in tower.items:
                ph=item.curr_dims[2]
                if item.tier==3 and z>0.5:    self.unplaced.append(item); continue
                if z+ph>self.H+self.EPS:      self.unplaced.append(item); continue
                if self._overlaps(bx,by,z,bl,bw,ph): self.unplaced.append(item); continue
                if supporter is not None:
                    ol=False; aw=item.weight; c=supporter
                    while c:
                        if c.current_load+aw>c.max_stack_load: ol=True; break
                        c=c.supporter
                    if ol: self.unplaced.append(item); continue

                item.pos=[bx,by,z]; item.curr_dims=[bl,bw,ph]
                item.supporter=supporter; item.is_door_single=False
                self.placed.append(item); self.current_weight+=item.weight
                c=supporter
                while c: c.current_load+=item.weight; c=c.supporter
                supporter=item; z+=ph

            free_rects=self._mr_update(free_rects,bx,by,bl,bw)
            placed_xs.append(bx+bl)

        if placed_xs: self._max_placed_x=max(placed_xs)

        # ── RETRY PASSES ──────────────────────────────────────────────────────
        retry=list(self.unplaced); self.unplaced=[]

        heavy_r   = [i for i in retry if i.weight >= TIER_HEAVY]
        medium_r  = [i for i in retry if TIER_MEDIUM <= i.weight < TIER_HEAVY]
        lt_tall_r = [i for i in retry if i.weight < TIER_MEDIUM and i.dims[2] > 90]
        lt_short_r= [i for i in retry if i.weight < TIER_MEDIUM and i.dims[2] <= 90]

        for item in sorted(heavy_r,  key=lambda i:(-i.weight,-i.dims[2])): self._place(item, floor_only=True)
        for item in sorted(medium_r, key=lambda i:(-i.weight,-i.dims[2])): self._place(item, stacking_only=True)
        still = list(self.unplaced); self.unplaced=[]
        for item in still: self._place(item, floor_only=True)
        for item in sorted(lt_tall_r, key=lambda i:(-i.weight,-i.dims[2])): self._place(item, stacking_only=True, prefer_heavy_base=True)
        still2=list(self.unplaced); self.unplaced=[]
        for item in still2: self._place(item)
        self._headroom_fill(lt_short_r)
        for item in doorstep: item.is_door_single=True; self._place(item, door_only=True)

    def _headroom_fill(self, short_items):
        """Fill headroom above stacked light items with short light items (3rd layer)."""
        if not short_items: return
        stacked_lights = [p for p in self.placed if p.pos[2]>0.5 and p.weight<TIER_MEDIUM]
        for base in sorted(stacked_lights, key=lambda p: -(p.pos[2]+p.curr_dims[2])):
            top_z   = base.pos[2]+base.curr_dims[2]
            headroom= self.H - top_z
            if headroom < 20: continue
            for item in list(short_items):
                if item.dims[2] > headroom: continue
                if item.weight > base.weight: continue
                l,w,h=item.dims[0],item.dims[1],item.dims[2]
                if not self._fp_match(l,w,base.curr_dims[0],base.curr_dims[1]):
                    if not self._fp_match(w,l,base.curr_dims[0],base.curr_dims[1]): continue
                    l,w=w,l
                x,y,z=base.pos[0],base.pos[1],top_z
                if self._overlaps(x,y,z,l,w,h): continue
                ok,sup=self._get_support(x,y,z,l,w,item)
                if not ok: continue
                if not self._validate_placement(item,x,y,z,l,w,h,sup): continue
                item.pos=[x,y,z]; item.curr_dims=[l,w,h]; item.supporter=sup
                self.placed.append(item); self.current_weight+=item.weight
                c=sup
                while c: c.current_load+=item.weight; c=c.supporter
                self._max_placed_x=max(self._max_placed_x,x+l)
                short_items.remove(item); break
        for item in sorted(short_items, key=lambda i:(i.weight,i.dims[2])): self._place(item, stacking_only=True)
        rem=list(self.unplaced); self.unplaced=[]
        for item in rem: self._place(item)

    # ── Pack entry point ──────────────────────────────────────────────────────
    def pack(self, items, use_block_loading=True):
        self.placed=[]; self.unplaced=[]; self.current_weight=0
        if not items: return
        sku_cnt = Counter(it.sku for it in items)
        fp_cnt  = Counter((min(it.dims[0],it.dims[1]),max(it.dims[0],it.dims[1])) for it in items)
        def _single(it):
            fp=(min(it.dims[0],it.dims[1]),max(it.dims[0],it.dims[1]))
            return sku_cnt[it.sku]==1 and fp_cnt[fp]==1
        all_d  = sorted(items, key=lambda i:(-i.dims[2],-i.tier,-i.weight,-i.vol))
        bulk   = [it for it in all_d if not _single(it)]
        single = [it for it in all_d if     _single(it)]
        if use_block_loading: self._block_pack(bulk)
        else:
            for item in bulk: self._place(item)
        self._compact_floor()
        self._singles_start_x=self._max_placed_x
        for item in single: self._place(item, use_block_loading, True)
        self._compact_floor()
        seen=set(); uniq=[]
        for item in self.placed:
            if id(item) in seen: continue
            seen.add(id(item)); uniq.append(item)
        self.placed=uniq

    def get_center_of_gravity(self):
        if not self.placed: return self.L/2,self.W/2,0.0
        tw=sum(i.weight for i in self.placed) or 1.0
        cx=sum((i.pos[0]+i.curr_dims[0]/2)*i.weight for i in self.placed)/tw
        cy=sum((i.pos[1]+i.curr_dims[1]/2)*i.weight for i in self.placed)/tw
        cz=sum((i.pos[2]+i.curr_dims[2]/2)*i.weight for i in self.placed)/tw
        return cx,cy,cz

    def utilization(self):
        cv=self.L*self.W*self.H
        pv=sum(i.curr_dims[0]*i.curr_dims[1]*i.curr_dims[2] for i in self.placed)
        vp=pv/cv*100 if cv else 0
        wp=self.current_weight/self.max_wt*100 if self.max_wt else 0
        bl=vp*0.65+wp*0.35
        return {"vol":round(vp,2),"wt":round(wp,2),"blended":round(bl,2)}

    def push_stragglers_to_door(self):
        if len(self.placed)<2: return
        fi=[p for p in self.placed if p.pos[2]<0.5]
        if not fi: return
        max_x=max(p.pos[0]+p.curr_dims[0] for p in fi)
        def _has_nbr(item):
            x1=item.pos[0]+item.curr_dims[0]
            for o in fi:
                if o is item: continue
                yov=(item.pos[1]<o.pos[1]+o.curr_dims[1]-0.5 and item.pos[1]+item.curr_dims[1]>o.pos[1]+0.5)
                xtch=(abs(item.pos[0]-(o.pos[0]+o.curr_dims[0]))<1.5 or abs(x1-o.pos[0])<1.5)
                xov=(item.pos[0]<o.pos[0]+o.curr_dims[0]-0.5 and x1>o.pos[0]+0.5)
                ytch=(abs(item.pos[1]-(o.pos[1]+o.curr_dims[1]))<1.5 or abs(item.pos[1]+item.curr_dims[1]-o.pos[1])<1.5)
                if (xtch and yov) or (ytch and xov): return True
            return False
        def _try_place(item,nx,yt):
            l,w,h=item.curr_dims; yt=max(0.0,min(yt,self.W-w)); E=self.EPS
            return (not any(nx+l>p.pos[0]+E and nx<p.pos[0]+p.curr_dims[0]-E and
                            yt+w>p.pos[1]+E and yt<p.pos[1]+p.curr_dims[1]-E and
                            h>p.pos[2]+E and 0<p.pos[2]+p.curr_dims[2]-E
                            for p in self.placed if p is not item)
                    and nx+l<=self.L+0.1)
        for item in list(fi):
            if item.pos[0]>=max_x-item.curr_dims[0]-2.0: continue
            if _has_nbr(item): continue
            l,w,_=item.curr_dims; mn=0.0
            for o in self.placed:
                if o is item: continue
                yov=(item.pos[1]<o.pos[1]+o.curr_dims[1]-0.1 and item.pos[1]+w>o.pos[1]+0.1)
                zov=(0.0<o.pos[2]+o.curr_dims[2]-0.1 and item.curr_dims[2]>o.pos[2]+0.1)
                if yov and zov and o.pos[0]+o.curr_dims[0]<=item.pos[0]+0.1:
                    mn=max(mn,o.pos[0]+o.curr_dims[0])
            nx=round(mn,1)
            if nx<item.pos[0]-1.0 and _try_place(item,nx,item.pos[1]):
                item.pos[0]=nx; continue
            cxs=sorted({round(p.pos[0]+p.curr_dims[0],1) for p in fi if p is not item}|{max_x})
            cxs=[c for c in cxs if c+l<=self.L+0.1]; moved=False
            for nx in sorted(cxs,reverse=True):
                for yt in [0.0,self.W-w,(self.W-w)/2.0,item.pos[1]]:
                    if _try_place(item,nx,yt):
                        item.pos=[nx,max(0.0,min(yt,self.W-w)),0.0]
                        item.is_door_single=True; moved=True; break
                if moved: break


# =============================================================================
#  DUMMY-CONTAINER VALIDATOR
# =============================================================================
class DummyValidator:
    """
    Step 1: Pack into a dummy container applying all physical rules.
    Step 2: Check utilisation is in [UTIL_TARGET_MIN, UTIL_TARGET_MAX].
    Step 3: If not, try up to MAX_ATTEMPTS repackings with different
            item orderings.
    Returns the best OptimizationEngine found.
    """
    MAX_ATTEMPTS = 6

    @staticmethod
    def validate_and_pack(c_name, items, use_bl, app_log_fn=None):
        best_eng  = None
        best_util = -1.0
        best_gap  = float("inf")

        sort_keys = [
            lambda lst: sorted(lst, key=lambda i: (-i.dims[2], -i.weight, -i.vol)),
            lambda lst: sorted(lst, key=lambda i: (-i.weight, -i.dims[2], -i.vol)),
            lambda lst: sorted(lst, key=lambda i: (-i.vol, -i.tier)),
            lambda lst: sorted(lst, key=lambda i: (-i.tier, -i.dims[2], i.weight)),
            lambda lst: sorted(lst, key=lambda i: (-i.dims[2], i.weight, -i.vol)),
            None,   # random shuffle
        ]

        for attempt, key_fn in enumerate(sort_keys):
            trial = [i.clone() for i in items]
            if key_fn is None:
                random.shuffle(trial)
            else:
                trial = key_fn(trial)

            eng = OptimizationEngine(c_name, CONTAINERS[c_name])
            eng.pack(trial, use_block_loading=use_bl)
            eng.push_stragglers_to_door()

            util = eng.utilization()["vol"]
            gap  = abs(util - (UTIL_TARGET_MIN + UTIL_TARGET_MAX)/2)

            # Prefer packing that fits all items AND is closest to target centre
            all_fit   = len(eng.unplaced) == 0
            in_target = UTIL_TARGET_MIN <= util <= UTIL_TARGET_MAX

            if best_eng is None:
                best_eng = eng; best_util = util; best_gap = gap

            # Priority: (1) all items fit, (2) in target range, (3) closest to centre
            prev_all_fit = len(best_eng.unplaced) == 0
            prev_in_tgt  = UTIL_TARGET_MIN <= best_util <= UTIL_TARGET_MAX

            better = False
            if all_fit and not prev_all_fit:
                better = True
            elif all_fit == prev_all_fit:
                if in_target and not prev_in_tgt:
                    better = True
                elif in_target == prev_in_tgt and gap < best_gap:
                    better = True

            if better:
                best_eng = eng; best_util = util; best_gap = gap

            if app_log_fn:
                app_log_fn(f"  Attempt {attempt+1}/{DummyValidator.MAX_ATTEMPTS}: "
                           f"vol={util:.1f}% placed={len(eng.placed)} unplaced={len(eng.unplaced)}")

            if in_target and all_fit:
                break   # perfect → stop early

        return best_eng


# =============================================================================
#  HOVER TOOLTIP
# =============================================================================
class PalletTooltip:
    def __init__(self, fig, ax, placed_items, c_map, tk_widget, container_dims):
        self.fig=fig; self.ax=ax; self.items=placed_items; self.c_map=c_map
        self.widget=tk_widget; self.container_dims=container_dims
        self._tip=None; self.selected_index=None
        fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        fig.canvas.mpl_connect('axes_leave_event',    lambda e: self._hide_and_clear())
        fig.canvas.mpl_connect('button_press_event',  self._on_click)

    def _hide_and_clear(self):
        self._hide()
        if self.selected_index is not None:
            self.selected_index=None; self._redraw()

    def _screen_bbox(self, item):
        x0,y0,z0=item.pos; l,w,h=item.curr_dims
        pts2d=[]; depths=[]; proj=self.ax.get_proj()
        for dx in (0,l):
            for dy in (0,w):
                for dz in (0,h):
                    try:
                        xd,yd,zd=proj3d.proj_transform(x0+dx,y0+dy,z0+dz,proj)
                        sx,sy=self.ax.transData.transform((xd,yd))
                        if np.isfinite(sx) and np.isfinite(sy):
                            pts2d.append((sx,sy)); depths.append(zd)
                    except: pass
        if len(pts2d)<2: return None,None
        xs=[p[0] for p in pts2d]; ys=[p[1] for p in pts2d]
        return (min(xs),min(ys),max(xs),max(ys)), min(depths)

    def _hit(self, ex, ey):
        bd=float("inf"); bi=-1
        for i,item in enumerate(self.items):
            bb,d=self._screen_bbox(item)
            if bb is None: continue
            x0,y0,x1,y1=bb
            if x0-4<=ex<=x1+4 and y0-4<=ey<=y1+4:
                if d<bd: bd=d; bi=i
        return bi

    def _on_move(self, event):
        if event.inaxes!=self.ax: self._hide_and_clear(); return
        idx=self._hit(event.x,event.y)
        if idx!=self.selected_index:
            self.selected_index=idx if idx!=-1 else None
            self._redraw()
            if self.selected_index is not None: self._show(self.items[self.selected_index],event)
            else: self._hide()

    def _redraw(self):
        self.ax.cla(); L,W,H=self.container_dims
        for xs,ys,zs in [
            ([0,L],[0,0],[0,0]),([0,L],[W,W],[0,0]),([0,L],[0,0],[H,H]),([0,L],[W,W],[H,H]),
            ([0,0],[0,W],[0,0]),([L,L],[0,W],[0,0]),([0,0],[0,W],[H,H]),([L,L],[0,W],[H,H]),
            ([0,0],[0,0],[0,H]),([L,L],[0,0],[0,H]),([0,0],[W,W],[0,H]),([L,L],[W,W],[0,H]),
        ]:
            self.ax.plot3D(xs,ys,zs,color="#5AAFFF",alpha=0.25,linewidth=0.6)
        for i,item in enumerate(self.items):
            x,y,z=item.pos; l,w,h=item.curr_dims
            col=self.c_map.get(item.sku,"#888"); is_door=getattr(item,"is_door_single",False)
            if i==self.selected_index:
                self.ax.bar3d(x,y,z,l,w,h,color=col,edgecolor="yellow",linewidth=1.5,alpha=1.0)
            else:
                av=0.85 if self.selected_index is not None else 0.92
                ec="#C8CEDD" if is_door else ("#FFFFFF" if z>0.5 else "#000000")
                self.ax.bar3d(x,y,z,l,w,h,color=col+"CC",edgecolor=ec,linewidth=0.8,alpha=av)
        self.ax.set_xlim(0,L); self.ax.set_ylim(0,W); self.ax.set_zlim(0,H)
        self.ax.set_box_aspect((L,W,H))
        for sp in self.ax.spines.values(): sp.set_visible(False)
        for pn in (self.ax.xaxis.pane,self.ax.yaxis.pane,self.ax.zaxis.pane):
            pn.fill=False; pn.set_edgecolor("#3A3F4A")
        self.ax.tick_params(colors="#C8CEDD",labelsize=7)
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes!=self.ax: return
        idx=self._hit(event.x,event.y)
        if idx!=-1: self._show(self.items[idx],event)

    def _show(self, item, event):
        self._hide()
        try:
            fig_h_px=self.fig.get_size_inches()[1]*self.fig.dpi
            rx=self.widget.winfo_rootx()+int(event.x)+18
            ry=self.widget.winfo_rooty()+int(fig_h_px-event.y)+18
        except: return
        t=tk.Toplevel(self.widget); t.overrideredirect(True)
        t.wm_geometry(f"+{rx}+{ry}")
        t.configure(bg="#1E2128",highlightbackground="#4A9EFF",highlightthickness=1)
        try:
            lp=item.current_load/item.max_stack_load*100 if item.max_stack_load else 0
            cl=max(0.0,item.max_stack_load-item.current_load)
            stk="Yes" if item.pos[2]>0.5 else "No"
            is_door=getattr(item,"is_door_single",False)
            tc={"Heavy":"#F26B6B","Medium":"#F5A623","Light":"#3ECF8E"}.get(item.tier_label,"#E8EAF0")
            cc="#3ECF8E" if lp<50 else ("#F5A623" if lp<85 else "#F26B6B")
            sc=self.c_map.get(item.sku,"#4A9EFF")
            layer=1; wk=item.supporter
            while wk: layer+=1; wk=wk.supporter
            rows=[
                ("Part Number",  item.sku,                          "#FFFFFF"),
                ("Shipment Ref", getattr(item,"shipment_ref","—"), "#5AAFFF"),
                ("Date",         getattr(item,"date","—"),          "#5AAFFF"),
                ("Destination",  getattr(item,"dest","—"),          "#5AAFFF"),
                ("Placement",    ">> Door" if is_door else "Main Block",
                                 "#FF4500" if is_door else "#9BA3B2"),
                ("Dims (cm)",    f"{item.curr_dims[0]:.0f}×{item.curr_dims[1]:.0f}×{item.curr_dims[2]:.0f}","#E8EAF0"),
                ("Weight",       f"{item.weight:.0f} kg",           "#E8EAF0"),
                ("Tier",         f"{item.tier_label} ({item.weight:.0f} kg)",tc),
                ("Stack Layer",  f"Layer {layer} of {MAX_STACK_LAYERS} max","#E8EAF0"),
                ("Pos X/Y/Z",    f"{item.pos[0]:.0f}/{item.pos[1]:.0f}/{item.pos[2]:.0f} cm","#E8EAF0"),
                ("Stacked",      stk, "#F5A623" if stk=="Yes" else "#9BA3B2"),
                ("Load on top",  f"{item.current_load:.0f} kg",     "#E8EAF0"),
                ("Cap left",     f"{cl:.0f} kg ({100-lp:.0f}% free)",cc),
            ]
            inner=tk.Frame(t,bg="#1E2128",padx=10,pady=8); inner.pack()
            tk.Frame(inner,bg=sc,width=3).grid(row=0,column=0,rowspan=len(rows)+1,sticky="ns",padx=(0,8))
            tk.Label(inner,text=f"  {item.sku}",bg="#1E2128",fg="#FFFFFF",
                     font=("Segoe UI",9,"bold")).grid(row=0,column=1,columnspan=2,sticky="w",pady=(0,5))
            for r,(lbl,val,fg) in enumerate(rows[1:],start=1):
                tk.Label(inner,text=lbl+":",bg="#1E2128",fg="#9BA3B2",
                         font=("Segoe UI",8),width=13,anchor="w").grid(row=r,column=1,sticky="w")
                tk.Label(inner,text=str(val),bg="#1E2128",fg=fg,
                         font=("Segoe UI",8,"bold"),anchor="w").grid(row=r,column=2,sticky="w",padx=(4,0))
            self._tip=t
        except Exception as e:
            print(f"Tooltip error: {e}"); t.destroy()

    def _hide(self):
        if self._tip:
            try: self._tip.destroy()
            except: pass
            self._tip=None


# =============================================================================
#  MANUAL ALLOCATION DIALOG
# =============================================================================
class ManualAllocationDialog:
    def __init__(self, parent_app):
        self.app = parent_app
        if not self.app.containers_used:
            messagebox.showwarning("No Containers","Run optimization first.",parent=self.app.root); return
        self.win = tk.Toplevel(self.app.root)
        self.win.title("Manual Pallet Allocation")
        self.win.geometry("1080x660"); self.win.minsize(900,560)
        self.win.configure(bg=self.app.C_BG); self.win.grab_set()
        self._orig_ids = {d["id"]:{id(i) for i in d["packer"].placed} for d in self.app.containers_used}
        self._working  = {d["id"]:list(d["packer"].placed)             for d in self.app.containers_used}
        self._staged_count=0
        self._build_ui(); self._refresh_combos()

    def _build_ui(self):
        C=self.app
        hdr=tk.Frame(self.win,bg="#161920",height=38); hdr.pack(fill=tk.X); hdr.pack_propagate(False)
        tk.Label(hdr,text="  ⇄  Manual Pallet Allocation",bg="#161920",fg=C.C_TEXT,font=("Segoe UI",10,"bold")).pack(side=tk.LEFT,pady=8)
        tk.Label(hdr,text="  Select pallets → choose destination → Move → Apply & Repack",bg="#161920",fg=C.C_TEXT2,font=("Segoe UI",8)).pack(side=tk.LEFT)
        body=tk.Frame(self.win,bg=C.C_BG); body.pack(fill=tk.BOTH,expand=True,padx=8,pady=6)

        def make_panel(side,color,title):
            outer=tk.Frame(body,bg=C.C_BG2); outer.pack(side=side,fill=tk.BOTH,expand=True)
            tk.Frame(outer,bg=color,height=3).pack(fill=tk.X)
            tk.Label(outer,text=f"  {title}",bg=C.C_BG3,fg=color,font=("Segoe UI",9,"bold"),padx=4,pady=5).pack(fill=tk.X)
            top=tk.Frame(outer,bg=C.C_BG2); top.pack(fill=tk.X,padx=6,pady=(4,2))
            var=tk.StringVar(); combo=ttk.Combobox(top,textvariable=var,state="readonly",style="Dark.TCombobox",width=32)
            combo.pack(side=tk.LEFT,padx=4)
            cnt=tk.Label(top,text="",bg=C.C_BG2,fg=C.C_TEXT3,font=("Segoe UI",8)); cnt.pack(side=tk.LEFT,padx=4)
            tf=tk.Frame(outer,bg=C.C_BG2); tf.pack(fill=tk.BOTH,expand=True,padx=6,pady=(0,6))
            tree=self._make_tree(tf)
            return var,combo,cnt,tree

        self.src_var,self.src_combo,self._src_count,self.src_tree = make_panel(tk.LEFT,C.C_BLUE,"SOURCE CONTAINER")

        mid=tk.Frame(body,bg=C.C_BG,width=88); mid.pack(side=tk.LEFT,fill=tk.Y,padx=4); mid.pack_propagate(False)
        tk.Frame(mid,bg=C.C_BG).pack(expand=True)
        def mbtn(txt,cmd,col="#3A3F4A"):
            b=tk.Button(mid,text=txt,bg=col,fg="white",font=("Segoe UI",8,"bold"),relief=tk.FLAT,padx=6,pady=8,cursor="hand2",command=cmd,wraplength=80)
            b.pack(pady=5,fill=tk.X,padx=4)
            b.bind("<Enter>",lambda e:b.config(bg=C.C_BLUE)); b.bind("<Leave>",lambda e:b.config(bg=col))
        mbtn("Move →\nto Dest",self._move_to_dest); mbtn("← Move\nto Src",self._move_to_src)
        tk.Frame(mid,bg=C.C_BORDER,height=1).pack(fill=tk.X,padx=6,pady=8)
        mbtn("Select All\nSrc",self._select_all_src); mbtn("Select All\nDst",self._select_all_dst)
        tk.Frame(mid,bg=C.C_BG).pack(expand=True)

        self.dst_var,self.dst_combo,self._dst_count,self.dst_tree = make_panel(tk.LEFT,C.C_GREEN,"DESTINATION CONTAINER")

        self.src_combo.bind("<<ComboboxSelected>>",self._on_src_change)
        self.dst_combo.bind("<<ComboboxSelected>>",self._on_dst_change)

        bot=tk.Frame(self.win,bg="#161920",height=46); bot.pack(fill=tk.X,side=tk.BOTTOM); bot.pack_propagate(False)
        self._status_lbl=tk.Label(bot,text="Ready",bg="#161920",fg=C.C_TEXT2,font=("Segoe UI",8)); self._status_lbl.pack(side=tk.LEFT,padx=12,pady=12)
        self._staged_lbl=tk.Label(bot,text="",bg="#161920",fg=C.C_AMBER,font=("Segoe UI",8,"bold")); self._staged_lbl.pack(side=tk.LEFT,padx=6)
        def abt(txt,cmd,col):
            b=tk.Button(bot,text=txt,bg=col,fg="white",font=("Segoe UI",9,"bold"),relief=tk.FLAT,padx=14,pady=8,cursor="hand2",command=cmd)
            b.pack(side=tk.RIGHT,padx=5,pady=8)
        abt("✕  Close",self.win.destroy,"#3A3F4A")
        self._apply_btn=tk.Button(bot,text="✓  Apply & Repack",bg="#1E7E4A",fg="white",font=("Segoe UI",9,"bold"),relief=tk.FLAT,padx=14,pady=8,cursor="hand2",command=self._apply)
        self._apply_btn.pack(side=tk.RIGHT,padx=5,pady=8)

    def _make_tree(self,parent):
        cols=("SKU","Dims (cm)","Wt kg","Tier","Stacked","Pos X/Y/Z")
        tv=ttk.Treeview(parent,columns=cols,show="headings",style="CT.Treeview",selectmode="extended")
        for col,w in zip(cols,[90,115,58,58,58,108]):
            tv.heading(col,text=col); tv.column(col,width=w,anchor=tk.CENTER,stretch=True)
        sb=ttk.Scrollbar(parent,orient=tk.VERTICAL,command=tv.yview,style="Dark.Vertical.TScrollbar")
        tv.configure(yscrollcommand=sb.set); tv.pack(side=tk.LEFT,fill=tk.BOTH,expand=True); sb.pack(side=tk.LEFT,fill=tk.Y)
        return tv

    def _combo_label(self,d):
        n=len(self._working.get(d["id"],[])); dest=d.get("dest","—"); date=d.get("date","—")
        return f"Container {d['id']}  ({d['c_name']})  {dest} / {str(date)[:10]}  [{n} pallets]"

    def _refresh_combos(self):
        lbs=[self._combo_label(d) for d in self.app.containers_used]
        self.src_combo["values"]=lbs; self.dst_combo["values"]=lbs

    def _get_container(self,combo_var):
        sel=combo_var.get()
        if not sel: return None
        vals=list(self.src_combo["values"])
        if sel not in vals: return None
        idx=vals.index(sel)
        if idx>=len(self.app.containers_used): return None
        return self.app.containers_used[idx]

    def _fill_tree(self,tree,items,cnt_lbl=None):
        tree.delete(*tree.get_children())
        for i,item in enumerate(items):
            stk="Yes" if item.pos[2]>0.5 else "No"
            dims=f"{item.curr_dims[0]:.0f}×{item.curr_dims[1]:.0f}×{item.curr_dims[2]:.0f}"
            pos=f"{item.pos[0]:.0f}/{item.pos[1]:.0f}/{item.pos[2]:.0f}"
            tree.insert("","end",iid=str(i),values=(item.sku,dims,f"{item.weight:.0f}",item.tier_label,stk,pos))
        if cnt_lbl: cnt_lbl.config(text=f"({len(items)} pallets)")

    def _on_src_change(self,_=None):
        d=self._get_container(self.src_var)
        if d: self._fill_tree(self.src_tree,self._working[d["id"]],self._src_count); self._refresh_combos()

    def _on_dst_change(self,_=None):
        d=self._get_container(self.dst_var)
        if d: self._fill_tree(self.dst_tree,self._working[d["id"]],self._dst_count); self._refresh_combos()

    def _select_all_src(self): self.src_tree.selection_set(self.src_tree.get_children())
    def _select_all_dst(self): self.dst_tree.selection_set(self.dst_tree.get_children())

    def _do_move(self,from_tree,from_var,to_var):
        sd=self._get_container(from_var); dd=self._get_container(to_var)
        if not sd or not dd: messagebox.showwarning("Select Containers","Please select both.",parent=self.win); return
        if sd["id"]==dd["id"]: messagebox.showwarning("Same Container","Must be different.",parent=self.win); return
        sel=from_tree.selection()
        if not sel: messagebox.showwarning("No Selection","Select pallets.",parent=self.win); return
        idxs=sorted([int(s) for s in sel],reverse=True)
        si=self._working[sd["id"]]; di=self._working[dd["id"]]; moved=[]
        for idx in idxs:
            if idx<len(si): moved.append(si.pop(idx))
        di.extend(moved); self._staged_count+=len(moved)
        self._staged_lbl.config(text=f"{self._staged_count} staged move(s)")
        self._status_lbl.config(text=f"Staged: {len(moved)} pallet(s) → Container {dd['id']}")
        self._refresh_combos()
        sl=self._combo_label(sd); dl=self._combo_label(dd)
        self.src_var.set(sl if sl in self.src_combo["values"] else "")
        self.dst_var.set(dl if dl in self.dst_combo["values"] else "")
        self._fill_tree(self.src_tree,self._working[sd["id"]],self._src_count)
        self._fill_tree(self.dst_tree,self._working[dd["id"]],self._dst_count)

    def _move_to_dest(self): self._do_move(self.src_tree,self.src_var,self.dst_var)
    def _move_to_src(self):  self._do_move(self.dst_tree,self.dst_var,self.src_var)

    def _apply(self):
        changes={}; any_change=False
        for d in self.app.containers_used:
            orig=self._orig_ids.get(d["id"],set()); wrk=self._working.get(d["id"],[])
            wset={id(i) for i in wrk}
            added=[i for i in wrk if id(i) not in orig]
            removed=[i for i in d["packer"].placed if id(i) not in wset]
            if added or removed: any_change=True
            changes[d["id"]]={"added":added,"removed":removed,"working":wrk}
        if not any_change: messagebox.showinfo("No Changes","No pallets were moved.",parent=self.win); return
        self._status_lbl.config(text="Validating…"); self._apply_btn.config(state=tk.DISABLED); self.win.update()
        errs=[]
        for d in self.app.containers_used:
            ch=changes[d["id"]]
            if not ch["added"]: continue
            test=self.app._pack_eng(d["c_name"],[i.clone() for i in ch["working"]])
            if test.unplaced:
                skus=", ".join(sorted({i.sku for i in test.unplaced}))
                errs.append(f"  • Container {d['id']}: {len(test.unplaced)} pallet(s) [{skus}] cannot fit.")
        if errs:
            messagebox.showwarning("⚠ Not Enough Space","Cannot complete:\n\n"+"\n".join(errs),parent=self.win)
            self._apply_btn.config(state=tk.NORMAL); self._status_lbl.config(text="Rejected — not enough space."); return
        to_remove=[]
        for d in self.app.containers_used:
            ch=changes[d["id"]]
            if not ch["working"]: to_remove.append(d); continue
            if not ch["added"] and not ch["removed"]: continue
            d["packer"]=self.app._pack_eng(d["c_name"],[i.clone() for i in ch["working"]])
        for d in to_remove: self.app.containers_used.remove(d)
        for i,d in enumerate(self.app.containers_used): d["id"]=i+1
        messagebox.showinfo("Done",f"✔  Allocation applied!\nContainers: {len(self.app.containers_used)}",parent=self.win)
        try: self.win.destroy()
        except: pass
        self.app.root.after(80,self._do_refresh)

    def _do_refresh(self):
        ni=min(self.app._cur_idx,max(0,len(self.app.containers_used)-1))
        self.app._cur_idx=ni; self.app._upd_cards(); self.app._refresh_nav()


# =============================================================================
#  MAIN APPLICATION
# =============================================================================
class LogiPackApp:
    C_BG="#1E2128"; C_BG2="#252830"; C_BG3="#2D3139"; C_BG4="#363B44"; C_BORDER="#3A3F4A"
    C_TEXT="#F0F2F8"; C_TEXT2="#C8CEDD"; C_TEXT3="#8A94A8"
    C_BLUE="#5AAFFF"; C_GREEN="#3ECF8E"; C_AMBER="#F5A623"; C_RED="#F26B6B"; C_SEL="#1A5FA8"

    def __init__(self, root):
        self.root=root; self.root.title("LogiPack Pro v5 — Container Loading Optimizer")
        self.root.geometry("1280x780"); self.root.minsize(1000,680); self.root.configure(bg=self.C_BG)
        self.items_data: list[Item]=[]; self.containers_used: list[dict]=[]
        self._tooltips=[]; self._cur_idx=0
        self._setup_styles(); self._build_ui()

    def _setup_styles(self):
        s=ttk.Style(); s.theme_use("clam")
        bg,bg2,bg3=self.C_BG,self.C_BG2,self.C_BG3; txt,txt2=self.C_TEXT,self.C_TEXT2
        s.configure("Dark.TFrame",background=bg)
        s.configure("TNotebook",background=bg2,borderwidth=0)
        s.configure("TNotebook.Tab",background=bg2,foreground=txt2,padding=[12,6],font=("Segoe UI",9))
        s.map("TNotebook.Tab",background=[("selected",bg)],foreground=[("selected",self.C_BLUE)])
        s.configure("CT.Treeview",background=bg2,fieldbackground=bg2,foreground=txt,rowheight=24,font=("Segoe UI",9),borderwidth=0)
        s.configure("CT.Treeview.Heading",background=bg3,foreground=txt2,font=("Segoe UI",8,"bold"),relief="flat")
        s.map("CT.Treeview",background=[("selected",self.C_SEL)],foreground=[("selected","#fff")])
        s.configure("Dark.Vertical.TScrollbar",background=bg3,troughcolor=bg2,arrowcolor=txt2,borderwidth=0)
        s.configure("Dark.TCombobox",background=bg3,foreground=txt,fieldbackground=bg3,arrowcolor=txt2,selectbackground=self.C_SEL,font=("Segoe UI",9))
        s.configure("Dark.TCheckbutton",background="#161920",foreground=txt2,font=("Segoe UI",9))
        s.map("Dark.TCheckbutton",background=[("active","#161920")],foreground=[("active",txt)])

    def _build_ui(self):
        self._build_titlebar(); self._build_toolbar()
        body=ttk.Frame(self.root,style="Dark.TFrame"); body.pack(fill=tk.BOTH,expand=True)
        self._build_center_panel(body); self._build_detail_panel(body); self._build_statusbar()

    def _build_titlebar(self):
        tb=tk.Frame(self.root,bg="#161920",height=36); tb.pack(fill=tk.X); tb.pack_propagate(False)
        tk.Label(tb,text="  LogiPack Pro v5",bg="#161920",fg=self.C_TEXT,font=("Segoe UI",10,"bold")).pack(side=tk.LEFT)
        tk.Label(tb,text=" — Container Loading Optimizer  |  Staircase · Physical Stack Rules · 85–95% Target",
                 bg="#161920",fg=self.C_TEXT2,font=("Segoe UI",8)).pack(side=tk.LEFT)

    def _build_toolbar(self):
        tb=tk.Frame(self.root,bg="#1a1d21",height=40); tb.pack(fill=tk.X); tb.pack_propagate(False)
        def tbtn(text,cmd,is_action=False,state=tk.NORMAL,color=None):
            bg_c=color if color else ("#1a1d21" if not is_action else self.C_SEL)
            b=tk.Button(tb,text=text,bg=bg_c,fg=self.C_TEXT2 if not is_action else "#fff",
                        activebackground=self.C_BG3,activeforeground=self.C_TEXT,
                        font=("Segoe UI",9),relief=tk.FLAT,bd=0,padx=10,pady=8,
                        cursor="hand2" if state==tk.NORMAL else "arrow",state=state,command=cmd)
            b.pack(side=tk.LEFT)
            if state==tk.NORMAL:
                b.bind("<Enter>",lambda e:b.config(bg=self.C_BG3))
                b.bind("<Leave>",lambda e:b.config(bg=bg_c))
            return b
        def sep(): tk.Frame(tb,bg=self.C_BORDER,width=1).pack(side=tk.LEFT,fill=tk.Y,padx=4,pady=6)
        tbtn("Open Excel",self._upload); sep()
        self._btn_opt =tbtn("Optimize",  self._run_multi,is_action=True)
        self._btn_stop=tbtn("Stop",      self._stop,     state=tk.DISABLED); sep()
        tk.Label(tb,text=" Manual:",bg="#1a1d21",fg=self.C_TEXT2,font=("Segoe UI",8)).pack(side=tk.LEFT)
        self.c_var=tk.StringVar(value="40GP")
        ttk.Combobox(tb,textvariable=self.c_var,values=list(CONTAINERS.keys()),width=6,state="readonly",style="Dark.TCombobox").pack(side=tk.LEFT,padx=3)
        tbtn("Pack Single",self._run_single); sep()
        self.block_var=tk.BooleanVar(value=True)
        ttk.Checkbutton(tb,text="Block Loading",variable=self.block_var,style="Dark.TCheckbutton").pack(side=tk.LEFT,padx=6); sep()
        tbtn("Export Excel",self._export_excel)
        tbtn("Export PDF",  self._export_pdf,state=tk.NORMAL if REPORTLAB_INSTALLED else tk.DISABLED)
        tbtn("Save",        self._save); sep()
        tbtn("⇄ Manual Alloc",self._manual_alloc,color="#2D3139")

    def _build_center_panel(self,parent):
        frame=tk.Frame(parent,bg=self.C_BG); frame.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
        cards=tk.Frame(frame,bg=self.C_BG2,height=76); cards.pack(fill=tk.X); cards.pack_propagate(False)
        self._sc={}
        for lbl,key,col in [("Total Containers","ct","Blue"),("Total Pallets","pallets",""),
                             ("Total Weight kg","weight",""),("Avg Vol Util %","util","Green"),
                             ("Avg Wt Util %","wt_util","Amber"),("Stacked Pallets","stacked","Amber")]:
            self._sc[key]=self._card(cards,"—",lbl,col)
        nav=tk.Frame(frame,bg=self.C_BG3,height=30); nav.pack(fill=tk.X); nav.pack_propagate(False)
        def nb(text,cmd):
            b=tk.Button(nav,text=text,bg=self.C_BG3,fg=self.C_TEXT2,activebackground=self.C_BG4,
                        activeforeground=self.C_TEXT,font=("Segoe UI",9),relief=tk.FLAT,bd=0,padx=10,pady=4,cursor="hand2",command=cmd)
            b.pack(side=tk.LEFT)
            b.bind("<Enter>",lambda e:b.config(bg=self.C_BG4)); b.bind("<Leave>",lambda e:b.config(bg=self.C_BG3))
        nb("◀ Prev",self._nav_prev)
        self._nav_lbl=tk.Label(nav,text="No containers loaded",bg=self.C_BG3,fg=self.C_TEXT,font=("Segoe UI",9,"bold")); self._nav_lbl.pack(side=tk.LEFT,padx=12)
        nb("Next ▶",self._nav_next)
        self.nb=ttk.Notebook(frame); self.nb.pack(fill=tk.BOTH,expand=True)
        self.tab_iso=ttk.Frame(self.nb,style="Dark.TFrame")
        self.tab_top=ttk.Frame(self.nb,style="Dark.TFrame")
        self.nb.add(self.tab_iso,text="  Isometric View  ")
        self.nb.add(self.tab_top,text="  Top View  ")

    def _card(self,parent,val,lbl,color=""):
        c=tk.Frame(parent,bg=self.C_BG3,padx=14,pady=8); c.pack(side=tk.LEFT,fill=tk.Y,expand=True,padx=4,pady=6)
        fg={"Blue":self.C_BLUE,"Green":self.C_GREEN,"Amber":self.C_AMBER}.get(color,self.C_TEXT)
        v=tk.Label(c,text=val,bg=self.C_BG3,fg=fg,font=("Segoe UI",18,"bold")); v.pack(anchor=tk.W)
        tk.Label(c,text=lbl,bg=self.C_BG3,fg=self.C_TEXT2,font=("Segoe UI",8)).pack(anchor=tk.W)
        return v

    def _build_detail_panel(self,parent):
        frame=tk.Frame(parent,bg=self.C_BG2,width=290); frame.pack(side=tk.RIGHT,fill=tk.Y); frame.pack_propagate(False)
        hdr=tk.Frame(frame,bg=self.C_BG3,height=28); hdr.pack(fill=tk.X)
        tk.Label(hdr,text="Container Details",bg=self.C_BG3,fg=self.C_TEXT,font=("Segoe UI",9,"bold"),padx=10).pack(side=tk.LEFT,pady=4)
        tbl=tk.Frame(frame,bg=self.C_BG2); tbl.pack(fill=tk.X,padx=8,pady=4)
        self._dr={}
        for key in ["Container ID","Container Size","Shipment Ref","Destination","Schedule Date",
                    "Dimensions (cm)","Max Capacity","Loaded Weight",
                    "Volume Util %","Weight Util %","Blended Util %","Pallets Loaded","Stacked","Remaining"]:
            row=tk.Frame(tbl,bg=self.C_BG2); row.pack(fill=tk.X,pady=1)
            tk.Label(row,text=key,bg=self.C_BG2,fg=self.C_TEXT2,font=("Segoe UI",8),width=18,anchor=tk.W).pack(side=tk.LEFT)
            v=tk.Label(row,text="—",bg=self.C_BG2,fg=self.C_TEXT,font=("Segoe UI",8,"bold"),anchor=tk.E); v.pack(side=tk.RIGHT)
            self._dr[key]=v
        tk.Frame(frame,bg=self.C_BORDER,height=1).pack(fill=tk.X,padx=8)
        br=tk.Frame(frame,bg=self.C_BG2); br.pack(fill=tk.X,pady=3)
        tk.Label(br,text="CoG:",bg=self.C_BG2,fg=self.C_TEXT2,font=("Segoe UI",8),padx=8).pack(side=tk.LEFT)
        self._bal=tk.Canvas(br,bg=self.C_BG4,height=10,highlightthickness=0); self._bal.pack(side=tk.LEFT,fill=tk.X,expand=True,padx=(0,4))
        self._ballbl=tk.Label(br,text="—",bg=self.C_BG2,fg=self.C_BLUE,font=("Segoe UI",8,"bold"),width=5); self._ballbl.pack(side=tk.LEFT)
        tk.Frame(frame,bg=self.C_BORDER,height=1).pack(fill=tk.X)
        tk.Label(frame,text="All Containers",bg=self.C_BG3,fg=self.C_TEXT,font=("Segoe UI",8,"bold"),padx=8).pack(fill=tk.X,pady=4)
        pw=tk.Frame(frame,bg=self.C_BG2); pw.pack(fill=tk.BOTH,expand=True)
        cl_cols=("#","Size","Destination","Pallets","Stacked","Vol%","Wt%")
        self.container_tree=ttk.Treeview(pw,columns=cl_cols,show="headings",style="CT.Treeview",selectmode="browse")
        for col,w in zip(cl_cols,[26,46,78,46,46,46,46]):
            self.container_tree.heading(col,text=col); self.container_tree.column(col,width=w,anchor=tk.CENTER,stretch=False)
        self.container_tree.tag_configure("hi",foreground=self.C_GREEN)
        self.container_tree.tag_configure("mid",foreground=self.C_AMBER)
        self.container_tree.tag_configure("lo",foreground=self.C_RED)
        sb=ttk.Scrollbar(pw,orient=tk.VERTICAL,command=self.container_tree.yview,style="Dark.Vertical.TScrollbar")
        self.container_tree.configure(yscrollcommand=sb.set)
        self.container_tree.pack(side=tk.LEFT,fill=tk.BOTH,expand=True); sb.pack(side=tk.LEFT,fill=tk.Y)
        self.container_tree.bind("<<TreeviewSelect>>",self._on_ct_sel)

    def _build_statusbar(self):
        sb=tk.Frame(self.root,bg="#161920",height=24); sb.pack(fill=tk.X,side=tk.BOTTOM); sb.pack_propagate(False)
        tk.Frame(sb,bg=self.C_GREEN,width=6,height=6).pack(side=tk.LEFT,padx=(10,4),pady=9)
        self._stlbl=tk.Label(sb,text="Ready",bg="#161920",fg=self.C_TEXT2,font=("Segoe UI",8)); self._stlbl.pack(side=tk.LEFT)
        tk.Label(sb,text="Creator - Kidzoro",bg="#161920",fg=self.C_TEXT2,font=("Segoe UI",8)).pack(side=tk.RIGHT,padx=14)
        self._tslbl=tk.Label(sb,text="",bg="#161920",fg=self.C_TEXT2,font=("Segoe UI",8)); self._tslbl.pack(side=tk.RIGHT,padx=14)
        self._tick()

    def _tick(self):
        self._tslbl.config(text=datetime.now().strftime("%d-%b-%Y  %H:%M"))
        self.root.after(30000,self._tick)

    def _log(self,msg,level="ok"):
        self._stlbl.config(text=msg,fg=self.C_GREEN if level=="ok" else self.C_RED)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def _nav_prev(self):
        if self.containers_used: self._goto(max(0,self._cur_idx-1))

    def _nav_next(self):
        if self.containers_used: self._goto(min(len(self.containers_used)-1,self._cur_idx+1))

    def _on_ct_sel(self,_):
        sel=self.container_tree.selection()
        if sel: self._goto(int(sel[0]),from_tree=True)

    def _goto(self,idx,from_tree=False):
        if not self.containers_used: return
        self._cur_idx=idx; d=self.containers_used[idx]; n=len(self.containers_used)
        dest=d.get("dest","—"); date=d.get("date","—")
        route=f"  {dest} / {date}" if dest not in ("Any","—") else ""
        self._nav_lbl.config(text=f"Container {idx+1} of {n}  ({d['c_name']}){route}")
        self._draw_views(d); self._fill_detail(d)
        if not from_tree:
            try: self.container_tree.selection_set(str(idx)); self.container_tree.see(str(idx))
            except: pass

    def _pop_container_tree(self):
        for c in self.container_tree.get_children(): self.container_tree.delete(c)
        for d in self.containers_used:
            p=d["packer"]; u=p.utilization()
            st=sum(1 for i in p.placed if i.pos[2]>0.5)
            dest=d.get("dest","—"); dd=dest[:10]+"…" if len(dest)>11 else dest
            tag="hi" if u["vol"]>=90 else ("mid" if u["vol"]>=70 else "lo")
            self.container_tree.insert("","end",iid=str(d["id"]-1),
                values=(d["id"],d["c_name"],dd,len(p.placed),st,f"{u['vol']:.0f}%",f"{u['wt']:.0f}%"),tags=(tag,))

    def _refresh_nav(self):
        n=len(self.containers_used); self._pop_container_tree()
        if n==0: self._nav_lbl.config(text="No containers loaded")
        else: self._cur_idx=min(self._cur_idx,n-1); self._goto(self._cur_idx)

    def _upload(self):
        path=filedialog.askopenfilename(title="Select Excel File",filetypes=[("Excel","*.xlsx *.xls"),("All","*.*")])
        if not path: return
        def sf(val):
            if pd.isna(val): return 0.0
            m=re.search(r"[-+]?\d*\.?\d+",str(val).replace(",",""))
            return float(m.group()) if m else 0.0
        try:
            df=pd.read_excel(path); self.items_data=[]; skipped=0
            for idx,r in df.iterrows():
                raw=r.get("Part Number",r.get("SKU",r.get("PART","")))
                sku=f"Item_{idx+1}" if pd.isna(raw) or str(raw).strip()=="" else str(raw).strip()
                l=sf(r.get("Length",r.get("L",0))); w=sf(r.get("Width",r.get("W",0)))
                h=sf(r.get("Height",r.get("H",0))); wt=sf(r.get("Weight",r.get("Gross Wt / Plt",r.get("Wt",0))))
                qty=int(sf(r.get("Qty",r.get("Quantity",1))) or 1)
                date_val=r.get("Schedule Date",r.get("Date","Any"))
                dest_val=r.get("Destination",  r.get("Location","Any"))
                ship_ref=r.get("Container",r.get("Shipment",r.get("Ref","—")))
                ship_ref=str(ship_ref) if not pd.isna(ship_ref) else "—"
                if l>0 and w>0 and h>0:
                    for _ in range(qty): self.items_data.append(Item(sku,l,w,h,wt,1,date_val,dest_val,ship_ref))
                else: skipped+=1
            self._log(f"Loaded {len(self.items_data)} pallets"+(f" ({skipped} skipped)" if skipped else ""))
            self._upd_cards()
            messagebox.showinfo("Loaded",f"✔  {len(self.items_data)} pallets loaded.\nClick Optimize.")
        except Exception as e:
            self._log(f"Error: {e}","err"); messagebox.showerror("Error",str(e))

    def _pack_eng(self, c_name, items):
        """Standard pack (no utilization retry — used for consolidation/manual alloc)."""
        eng=OptimizationEngine(c_name,CONTAINERS[c_name])
        eng.pack(items,use_block_loading=self.block_var.get())
        eng.push_stragglers_to_door()
        return eng

    def _pack_with_target(self, c_name, items):
        """
        Dummy-container validation pack:
        1. Try multiple orderings to hit UTIL_TARGET_MIN – UTIL_TARGET_MAX.
        2. Returns the best OptimizationEngine found.
        """
        return DummyValidator.validate_and_pack(
            c_name, items, self.block_var.get(),
            app_log_fn=lambda m: self.root.after(0, lambda mm=m: self._log(mm))
        )

    def _post_consolidate(self):
        MIN_VOL=85.0; MIN_WT=70.0
        changed=True; passes=0
        while changed and passes<25:
            changed=False; passes+=1
            by_route=defaultdict(list)
            for d in self.containers_used:
                by_route[(d.get("dest","—"),d.get("date","—"))].append(d)
            for route,group in by_route.items():
                if len(group)<2: continue
                group_sorted=sorted(group,key=lambda d:len(d["packer"].placed))
                small=group_sorted[0]; us=small["packer"].utilization()
                if us["vol"]>=MIN_VOL or us["wt"]>=MIN_WT: continue
                merged=False
                for other in group_sorted[1:]:
                    all_items=([i.clone() for i in other["packer"].placed]+
                               [i.clone() for i in small["packer"].placed])
                    for cn in ([other["c_name"]]+[x for x in AUTO_CONTAINER_SIZES if x!=other["c_name"]]):
                        retry=self._pack_eng(cn,all_items)
                        if not retry.unplaced:
                            msg=f"Consolidated Ctr {small['id']} (vol {us['vol']:.0f}%) → Ctr {other['id']} ({cn})"
                            self.root.after(0,lambda m=msg:self._log(m))
                            other["packer"]=retry; other["c_name"]=cn; other["c_info"]=CONTAINERS[cn]
                            self.containers_used.remove(small)
                            for i,d in enumerate(self.containers_used): d["id"]=i+1
                            changed=True; merged=True; break
                    if merged: break
                if changed: break

    def _run_single(self):
        if not self.items_data: messagebox.showwarning("No Data","Load Excel first."); return
        self._clear(); c=self.c_var.get()
        eng=self._pack_with_target(c,[i.clone() for i in self.items_data])
        st=sum(1 for i in eng.placed if i.pos[2]>0.5)
        self.containers_used.append({"id":1,"c_name":c,"packer":eng,"c_info":CONTAINERS[c],"date":"Any","dest":"Any"})
        self._upd_cards(); self._cur_idx=0; self._refresh_nav()
        u=eng.utilization()
        self._log(f"Done — {len(eng.placed)} placed, {st} stacked, {len(eng.unplaced)} left | vol {u['vol']:.0f}% wt {u['wt']:.0f}%")

    def _run_multi(self):
        if not self.items_data: messagebox.showwarning("No Data","Load Excel first."); return
        self._clear(); self._log("Optimization started…")
        self._btn_opt.config(state=tk.DISABLED); self._btn_stop.config(state=tk.NORMAL)
        self._stop_flag=False; threading.Thread(target=self._multi_thread,daemon=True).start()

    def _stop(self): self._stop_flag=True; self._log("Stopped by user.","err")

    def _multi_thread(self):
        groups=defaultdict(list)
        for item in self.items_data: groups[(item.dest,item.date)].append(item)
        sorted_groups=sorted(groups.items(),key=lambda kv:-len(kv[1]))
        c_num=1
        for (grp_dest,grp_date),grp_items in sorted_groups:
            if getattr(self,"_stop_flag",False): break
            route=f"{grp_dest} / {grp_date}"
            self.root.after(0,lambda r=route:self._log(f"Routing: {r}"))
            remaining=[i.clone() for i in grp_items]

            while remaining and not getattr(self,"_stop_flag",False):
                needs_hc=any(i.dims[2]>CONTAINERS["20GP"]["H"] for i in remaining)
                sizes_to_try=["40HC"] if needs_hc else AUTO_CONTAINER_SIZES

                # ── DUMMY CONTAINER VALIDATION + UTILIZATION TARGET ───────────
                fits={}
                for cn in sizes_to_try:
                    fits[cn]=self._pack_with_target(cn,[i.clone() for i in remaining])

                scored=[]
                for cn,eng in fits.items():
                    if not eng.placed: continue
                    u=eng.utilization()
                    scored.append((len(eng.unplaced),-u["blended"],cn,eng))
                if scored: scored.sort(); _,_,best_n,best_p=scored[0]
                else:
                    self.root.after(0,lambda:messagebox.showerror("Error","No item fits any container.")); break
                if not best_p.placed:
                    self.root.after(0,lambda:messagebox.showerror("Error","No item fits any container.")); break

                # ── Aggressive consolidation into existing containers ──────────
                consolidated=False
                same_route=[d for d in self.containers_used[-30:]
                            if d.get("dest")==grp_dest and d.get("date")==grp_date]
                for prev in reversed(same_route):
                    combined=[i.clone() for i in prev["packer"].placed]+[i.clone() for i in remaining]
                    for cn in ([prev["c_name"]]+[x for x in AUTO_CONTAINER_SIZES if x!=prev["c_name"]]):
                        retry=self._pack_with_target(cn,combined)
                        if not retry.unplaced:
                            prev["packer"]=retry; prev["c_name"]=cn; prev["c_info"]=CONTAINERS[cn]
                            remaining=[]; consolidated=True
                            msg=f"Container {prev['id']} consolidated ({cn}) → {len(retry.placed)} pallets"
                            self.root.after(0,lambda m=msg:self._log(m))
                            self.root.after(0,lambda:(self._upd_cards(),self._refresh_nav()))
                            break
                    if consolidated: break
                if consolidated: break

                st=sum(1 for i in best_p.placed if i.pos[2]>0.5)
                u=best_p.utilization()
                in_range=" ✓" if UTIL_TARGET_MIN<=u["vol"]<=UTIL_TARGET_MAX else " ⚠"
                msg=f"Container {c_num} ({best_n}) [{grp_dest}]: {len(best_p.placed)} pallets, {st} stacked | vol {u['vol']:.0f}%{in_range} wt {u['wt']:.0f}%"
                self.root.after(0,lambda m=msg:self._log(m))
                self.containers_used.append({"id":c_num,"c_name":best_n,"packer":best_p,
                                             "c_info":CONTAINERS[best_n],"date":grp_date,"dest":grp_dest})
                remaining=best_p.unplaced; c_num+=1
                self.root.after(0,lambda:(self._upd_cards(),self._refresh_nav()))

        if not getattr(self,"_stop_flag",False):
            self.root.after(0,lambda:self._log("Running post-consolidation pass…"))
            self._post_consolidate()
        self.root.after(0,self._done)

    def _done(self):
        self._btn_opt.config(state=tk.NORMAL); self._btn_stop.config(state=tk.DISABLED)
        n=len(self.containers_used)
        ts=sum(sum(1 for i in d["packer"].placed if i.pos[2]>0.5) for d in self.containers_used)
        vols=[d["packer"].utilization()["vol"] for d in self.containers_used]
        avg_v=round(sum(vols)/len(vols),1) if vols else 0
        self._log(f"Complete — {n} container(s), {ts} stacked, avg vol {avg_v}%.")
        self._upd_cards(); self._cur_idx=0; self._refresh_nav()
        messagebox.showinfo("Done",f"✔  {n} containers used.\n{ts} pallets stacked.\nAvg vol utilisation: {avg_v}%")

    def _clear(self):
        self.containers_used=[]; self._tooltips=[]; self._cur_idx=0
        for tab in (self.tab_iso,self.tab_top):
            for w in tab.winfo_children(): w.destroy()
        for c in self.container_tree.get_children(): self.container_tree.delete(c)
        self._nav_lbl.config(text="No containers loaded")

    def _upd_cards(self):
        if not self.containers_used:
            for k in self._sc: self._sc[k].config(text="—")
            self._sc["pallets"].config(text=str(len(self.items_data))); return
        n=len(self.containers_used)
        tp=sum(len(d["packer"].placed) for d in self.containers_used)
        tw=sum(d["packer"].current_weight for d in self.containers_used)
        ts=sum(sum(1 for i in d["packer"].placed if i.pos[2]>0.5) for d in self.containers_used)
        vus=[d["packer"].utilization()["vol"] for d in self.containers_used]
        wus=[d["packer"].utilization()["wt"]  for d in self.containers_used]
        av=round(sum(vus)/len(vus),1) if vus else 0
        aw=round(sum(wus)/len(wus),1) if wus else 0
        self._sc["ct"].config(text=str(n)); self._sc["pallets"].config(text=f"{tp:,}")
        self._sc["weight"].config(text=f"{tw:,.0f}")
        self._sc["util"].config(text=f"{av}%"); self._sc["wt_util"].config(text=f"{aw}%")
        self._sc["stacked"].config(text=f"{ts:,}")

    def _manual_alloc(self): ManualAllocationDialog(self)

    def _draw_views(self,data):
        self._tooltips=[]
        for tab in (self.tab_iso,self.tab_top):
            for w in tab.winfo_children(): w.destroy()
        lbl=f"Container {data['id']} ({data['c_name']})"
        self._draw_one(self.tab_iso,data,25,-60,f"{lbl} — Isometric",tooltip=True)
        self._draw_one(self.tab_top,data,90,-90,f"{lbl} — Top View",tooltip=False)

    def _draw_one(self,tab,data,elev,azim,title,tooltip=True):
        packer=data["packer"]; c=data["c_info"]; bg=self.C_BG
        fig=plt.Figure(figsize=(9,5.2),facecolor=bg)
        ax=fig.add_subplot(111,projection="3d"); ax.set_facecolor(bg); fig.patch.set_facecolor(bg)
        L,W,H=c["L"],c["W"],c["H"]
        for xs,ys,zs in [([0,L],[0,0],[0,0]),([0,L],[W,W],[0,0]),([0,L],[0,0],[H,H]),([0,L],[W,W],[H,H]),
                         ([0,0],[0,W],[0,0]),([L,L],[0,W],[0,0]),([0,0],[0,W],[H,H]),([L,L],[0,W],[H,H]),
                         ([0,0],[0,0],[0,H]),([L,L],[0,0],[0,H]),([0,0],[W,W],[0,H]),([L,L],[W,W],[0,H])]:
            ax.plot3D(xs,ys,zs,color=self.C_BLUE,alpha=0.25,linewidth=0.6)
        skus=sorted({i.sku for i in packer.placed})
        c_map={s:SKU_COLORS[n%len(SKU_COLORS)] for n,s in enumerate(skus)}
        for item in sorted(packer.placed,key=lambda i:-(i.pos[0]+i.pos[1]+i.pos[2])):
            x,y,z=item.pos; dl,dw,dh=item.curr_dims; col=c_map.get(item.sku,"#888")
            is_door=getattr(item,"is_door_single",False)
            edge="#FFFFFF"; lw=0.8
            if not is_door and z<0.5: edge="#000000"; lw=0.3
            ax.bar3d(x,y,z,dl,dw,dh,color=col+"CC",edgecolor=edge,alpha=0.92,linewidth=lw)
        ax.set_xlim(0,L); ax.set_ylim(0,W); ax.set_zlim(0,H); ax.set_box_aspect((L,W,H)); ax.view_init(elev=elev,azim=azim)
        st=sum(1 for i in packer.placed if i.pos[2]>0.5); u=packer.utilization()
        in_r=" ✓" if UTIL_TARGET_MIN<=u["vol"]<=UTIL_TARGET_MAX else " ⚠"
        ax.set_title(f"{title}  |  {st} stacked  |  vol {u['vol']:.0f}%{in_r}  wt {u['wt']:.0f}%",
                     color=self.C_TEXT2,fontsize=9,pad=8)
        for sp in ax.spines.values(): sp.set_visible(False)
        for pn in (ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane):
            pn.fill=False; pn.set_edgecolor(self.C_BORDER)
        ax.tick_params(colors=self.C_TEXT2,labelsize=7)
        patches=[mpatches.Patch(color=c_map[s],label=f"{s} (SKU)") for s in list(c_map)[:8]]
        if patches:
            ax.legend(handles=patches,loc="upper left",fontsize=7,ncol=2,facecolor=self.C_BG3,edgecolor=self.C_BORDER,labelcolor=self.C_TEXT)
        canvas=FigureCanvasTkAgg(fig,master=tab)
        canvas.get_tk_widget().configure(bg=bg); canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True); canvas.draw()
        if tooltip and packer.placed:
            tip=PalletTooltip(fig,ax,packer.placed,c_map,canvas.get_tk_widget(),container_dims=(L,W,H))
            self._tooltips.append(tip)

    def _make_iso_fig(self,data):
        packer=data["packer"]; c=data["c_info"]; bg=self.C_BG
        fig=plt.Figure(figsize=(10,5),facecolor=bg); ax=fig.add_subplot(111,projection="3d")
        ax.set_facecolor(bg); fig.patch.set_facecolor(bg); L,W,H=c["L"],c["W"],c["H"]
        for xs,ys,zs in [([0,L],[0,0],[0,0]),([0,L],[W,W],[0,0]),([0,L],[0,0],[H,H]),([0,L],[W,W],[H,H]),
                         ([0,0],[0,W],[0,0]),([L,L],[0,W],[0,0]),([0,0],[0,W],[H,H]),([L,L],[0,W],[H,H]),
                         ([0,0],[0,0],[0,H]),([L,L],[0,0],[0,H]),([0,0],[W,W],[0,H]),([L,L],[W,W],[0,H])]:
            ax.plot3D(xs,ys,zs,color="#5AAFFF",alpha=0.3,linewidth=0.7)
        skus=sorted({i.sku for i in packer.placed})
        c_map={s:SKU_COLORS[n%len(SKU_COLORS)] for n,s in enumerate(skus)}
        for item in sorted(packer.placed,key=lambda i:-(i.pos[0]+i.pos[1]+i.pos[2])):
            x,y,z=item.pos; dl,dw,dh=item.curr_dims; col=c_map.get(item.sku,"#888")
            is_door=getattr(item,"is_door_single",False); edge="#FFFFFF"; lw=0.8
            if not is_door and z<0.5: edge="#000000"; lw=0.3
            ax.bar3d(x,y,z,dl,dw,dh,color=col+"CC",edgecolor=edge,alpha=0.92,linewidth=lw)
        ax.set_xlim(0,L); ax.set_ylim(0,W); ax.set_zlim(0,H); ax.set_box_aspect((L,W,H)); ax.view_init(elev=25,azim=-60)
        st=sum(1 for i in packer.placed if i.pos[2]>0.5); u=packer.utilization()
        ax.set_title(f"Container {data['id']} ({data['c_name']})  |  {st} stacked  |  vol {u['vol']:.0f}% / wt {u['wt']:.0f}%",color="white",fontsize=10,pad=8)
        for sp in ax.spines.values(): sp.set_visible(False)
        for pn in (ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane): pn.fill=False; pn.set_edgecolor("#3A3F4A")
        ax.tick_params(colors="#CCCCCC",labelsize=7)
        patches=[mpatches.Patch(color=c_map[s],label=f"{s} (SKU)") for s in list(c_map)[:10]]
        if patches: ax.legend(handles=patches,loc="upper left",fontsize=7,ncol=2,facecolor="#2D3139",edgecolor="#3A3F4A",labelcolor="white")
        return fig

    def _fill_detail(self,data):
        p=data["packer"]; c=data["c_info"]; u=p.utilization()
        st=sum(1 for i in p.placed if i.pos[2]>0.5); ds=sum(1 for i in p.placed if getattr(i,"is_door_single",False)); rm=len(p.unplaced)
        srefs=sorted({getattr(i,"shipment_ref","—") for i in p.placed if getattr(i,"shipment_ref","—")!="—"})
        vals={"Container ID":f"Container {data['id']}","Container Size":data["c_name"],
              "Shipment Ref":"/".join(srefs) if srefs else "—",
              "Destination":data.get("dest","—"),"Schedule Date":data.get("date","—"),
              "Dimensions (cm)":f"{c['L']:.0f}×{c['W']:.0f}×{c['H']:.0f}",
              "Max Capacity":f"{c['MaxWt']:,} kg","Loaded Weight":f"{p.current_weight:,.0f} kg",
              "Volume Util %":f"{u['vol']:.1f}%","Weight Util %":f"{u['wt']:.1f}%","Blended Util %":f"{u['blended']:.1f}%",
              "Pallets Loaded":str(len(p.placed)),"Stacked":str(st),
              "Remaining":f"{rm}  ({ds} door-side)" if ds else str(rm)}
        for k,v in vals.items():
            lbl=self._dr[k]; lbl.config(text=v,fg=self.C_TEXT)
            if k in ("Volume Util %","Weight Util %","Blended Util %"):
                pv=float(v.replace("%",""))
                lbl.config(fg=(self.C_GREEN if pv>=90 else (self.C_AMBER if pv>=70 else self.C_RED)))
            elif k=="Remaining": lbl.config(fg=self.C_GREEN if rm==0 else self.C_RED)
            elif k=="Stacked":   lbl.config(fg=self.C_AMBER if st>0 else self.C_TEXT3)
        cg_x,_,_=p.get_center_of_gravity(); pct=cg_x/c["L"]*100 if c["L"] else 50
        self._ballbl.config(text=f"{pct:.0f}%",fg=self.C_GREEN if 40<=pct<=60 else self.C_RED)
        self._bal.update_idletasks(); bw=self._bal.winfo_width() or 120
        self._bal.delete("all")
        self._bal.create_rectangle(0,0,int(bw*pct/100),10,fill=self.C_BLUE,outline="")
        self._bal.create_line(bw//2,0,bw//2,10,fill="white",width=1)

    def _export_excel(self):
        if not self.containers_used: messagebox.showwarning("Nothing","Run optimization first."); return
        path=filedialog.asksaveasfilename(defaultextension=".xlsx",filetypes=[("Excel","*.xlsx")])
        if not path: return
        rows=[]
        for d in self.containers_used:
            cg_x,cg_y,cg_z=d["packer"].get_center_of_gravity(); u=d["packer"].utilization()
            for item in d["packer"].placed:
                layer=1; wk=item.supporter
                while wk: layer+=1; wk=wk.supporter
                rows.append({"Container_ID":d["id"],"Type":d["c_name"],
                             "Shipment_Ref":getattr(item,"shipment_ref","—"),
                             "Destination":d.get("dest","—"),"Schedule_Date":d.get("date","—"),
                             "Vol_Util_pct":u["vol"],"Wt_Util_pct":u["wt"],
                             "SKU":item.sku,"Weight_kg":item.weight,"Tier":item.tier_label,
                             "Length_cm":item.curr_dims[0],"Width_cm":item.curr_dims[1],"Height_cm":item.curr_dims[2],
                             "Pos_X":round(item.pos[0],1),"Pos_Y":round(item.pos[1],1),"Pos_Z":round(item.pos[2],1),
                             "Stack_Layer":layer,"Stacked":"Yes" if item.pos[2]>0.5 else "No",
                             "Load_on_top_kg":round(item.current_load,1),
                             "Capacity_remaining_kg":round(max(0,item.max_stack_load-item.current_load),1),
                             "CoG_X":round(cg_x,1),"CoG_Y":round(cg_y,1),"CoG_Z":round(cg_z,1)})
        pd.DataFrame(rows).to_excel(path,index=False)
        self._log(f"Exported: {os.path.basename(path)}")
        messagebox.showinfo("Done",f"Saved:\n{os.path.basename(path)}")

    def _export_pdf(self):
        if not REPORTLAB_INSTALLED: messagebox.showerror("Missing","pip install reportlab"); return
        if not self.containers_used: messagebox.showwarning("Nothing","Run optimization first."); return
        path=filedialog.asksaveasfilename(defaultextension=".pdf",filetypes=[("PDF","*.pdf")])
        if not path: return
        self._log("Generating PDF…")
        doc=SimpleDocTemplate(path,pagesize=A4,leftMargin=1.5*cm,rightMargin=1.5*cm,topMargin=1.5*cm,bottomMargin=1.5*cm)
        st=getSampleStyleSheet(); els=[]
        els.append(Paragraph("LogiPack Pro v5 — Load Manifest",st["Title"]))
        els.append(Paragraph(datetime.now().strftime("%Y-%m-%d  %H:%M"),st["Normal"]))
        els.append(Spacer(1,12))
        for d in self.containers_used:
            p=d["packer"]; c=d["c_info"]
            cg_x,_,_=p.get_center_of_gravity(); axle=round(cg_x/c["L"]*100,1) if c["L"] else 50
            stk=sum(1 for i in p.placed if i.pos[2]>0.5); u=p.utilization()
            dest=d.get("dest","—"); date=d.get("date","—")
            srefs=sorted({getattr(i,"shipment_ref","—") for i in p.placed if getattr(i,"shipment_ref","—")!="—"})
            sr_str=f" [{'/'.join(srefs)}]" if srefs else ""
            els.append(Paragraph(f"Container {d['id']} ({d['c_name']}){sr_str}  —  {dest}  /  {date}",st["Heading2"]))
            els.append(Paragraph(
                f"Pallets: {len(p.placed)} ({stk} stacked)  |  Weight: {p.current_weight:,.0f}/{c['MaxWt']:,} kg  |  "
                f"Vol: {u['vol']:.1f}%  Wt: {u['wt']:.1f}%  Blended: {u['blended']:.1f}%  |  CoG: {axle}%",st["Normal"]))
            els.append(Spacer(1,6))
            try:
                iso_fig=self._make_iso_fig(d); buf=io.BytesIO()
                iso_fig.savefig(buf,format="png",dpi=110,facecolor=iso_fig.get_facecolor(),bbox_inches="tight")
                plt.close(iso_fig); buf.seek(0)
                img_w=A4[0]-3*cm; img_h=img_w*0.48
                els.append(RLImage(buf,width=img_w,height=img_h))
            except Exception as e:
                els.append(Paragraph(f"[3D error: {e}]",st["Normal"]))
            els.append(Spacer(1,8))
            tbl_rows=[["SKU","Qty","Total Wt","Dims (cm)","Tier","Stacked"]]
            for sku,cnt in sorted(Counter(i.sku for i in p.placed).items()):
                s=next(i for i in p.placed if i.sku==sku)
                stk_c=sum(1 for i in p.placed if i.sku==sku and i.pos[2]>0.5)
                tbl_rows.append([sku,str(cnt),f"{s.weight*cnt:,.0f} kg",
                                 f"{s.dims[0]:.0f}×{s.dims[1]:.0f}×{s.dims[2]:.0f}",s.tier_label,f"{stk_c}/{cnt}"])
            t=Table(tbl_rows,colWidths=[3.5*cm,1.2*cm,2.5*cm,3.2*cm,2*cm,2*cm])
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1A5FA8")),
                                   ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                                   ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                                   ("FONTSIZE",(0,0),(-1,-1),8),("ALIGN",(0,0),(-1,-1),"CENTER"),
                                   ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F4F6FA")]),
                                   ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CCC")),
                                   ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
            els.append(t); els.append(Spacer(1,20))
        doc.build(els)
        self._log(f"PDF saved: {os.path.basename(path)}")
        messagebox.showinfo("Done",f"PDF saved:\n{os.path.basename(path)}")

    def _save(self):
        if not self.items_data: messagebox.showwarning("Nothing","Load data first."); return
        path=filedialog.asksaveasfilename(defaultextension=".xlsx",filetypes=[("Excel","*.xlsx")])
        if not path: return
        pd.DataFrame([{"Part Number":i.sku,"Length":i.dims[0],"Width":i.dims[1],"Height":i.dims[2],
                       "Weight":i.weight,"Schedule Date":i.date,"Destination":i.dest}
                      for i in self.items_data]).to_excel(path,index=False)
        self._log(f"Saved: {os.path.basename(path)}"); messagebox.showinfo("Saved","Project saved.")


# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    LogiPackApp(root); root.mainloop()

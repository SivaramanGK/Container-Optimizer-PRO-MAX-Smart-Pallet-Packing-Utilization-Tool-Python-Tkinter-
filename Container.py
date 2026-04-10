import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D, proj3d 
import itertools
from collections import Counter
#main code
# --- Container Constants (cm) ---
CONTAINERS = {
    "20GP": {"L": 589.8, "W": 235.2, "H": 239.3},
    "40GP": {"L": 1203.2, "W": 235.2, "H": 239.3},
    "40HC": {"L": 1203.2, "W": 235.2, "H": 269.8}
}

class Item:
    def __init__(self, sku, l, w, h, weight=0.0):
        self.sku = sku
        self.dims = [l, w, h]
        self.vol = l * w * h
        self.weight = weight
        self.pos = [0, 0, 0]
        self.curr_dims = [l, w, h]

class OptimizationEngine:
    def __init__(self, c_dims):
        door_clearance = 2.0 
        self.L = c_dims[0] - door_clearance
        self.W = c_dims[1]
        self.H = c_dims[2]
        self.placed = []
        self.unplaced = []

    def intersects(self, x, y, z, l, w, h):
        tol = 0.1 
        if x + l > self.L + tol or y + w > self.W + tol or z + h > self.H + tol:
            return True
            
        for p in self.placed:
            px, py, pz = p.pos
            pl, pw, ph = p.curr_dims
            if not (x + l <= px + tol or x >= px + pl - tol or 
                    y + w <= py + tol or y >= py + pw - tol or 
                    z + h <= pz + tol or z >= pz + ph - tol):
                return True
        return False

    def pack(self, items):
        if not items:
            return

        # Find the maximum weight to create our Tiers
        max_wt = max((i.weight for i in items), default=1)
        if max_wt == 0: 
            max_wt = 1

        # Creates 3 Weight Classes: 3=Heavy, 2=Medium, 1=Light
        def get_tier(wt):
            if wt >= max_wt * 0.66: return 3
            if wt >= max_wt * 0.33: return 2
            return 1

        # --- THE FIX: HYBRID SORTING ---
        # 1. Sorts by Weight Tier (Ensures Heavy goes on bottom, Light on top)
        # 2. Sorts by Volume (Restores the perfect 48-pallet Tetris puzzle logic!)
        items.sort(key=lambda x: (get_tier(x.weight), x.vol), reverse=True)
        
        for item in items:
            best_pt = None
            best_dims = None
            min_score = float('inf')

            xs = {0.0}
            ys = {0.0}
            zs = {0.0}
            
            for p in self.placed:
                xs.add(round(p.pos[0] + p.curr_dims[0], 2))
                ys.add(round(p.pos[1] + p.curr_dims[1], 2))
                zs.add(round(p.pos[2] + p.curr_dims[2], 2))
            
            pts = [[x, y, z] for x in xs for y in ys for z in zs 
                   if x < self.L and y < self.W and z < self.H]

            for pt in pts:
                orientations = [(item.dims[0], item.dims[1], item.dims[2]),
                               (item.dims[1], item.dims[0], item.dims[2])]
                
                for dims in orientations:
                    if not self.intersects(pt[0], pt[1], pt[2], dims[0], dims[1], dims[2]):
                        
                        score = (pt[2] + dims[2]) * 1000000 + (pt[0] + dims[0]) * 1000 + (pt[1] + dims[1])
                        
                        if score < min_score:
                            min_score = score
                            best_pt, best_dims = pt, dims
                            
            if best_pt:
                item.pos, item.curr_dims = best_pt, best_dims
                self.placed.append(item)
            else:
                self.unplaced.append(item)

class ProfessionalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LogiSmart  - EDC")
        self.root.geometry("1300x850")
        self.items_data = []
        self.packer = None
        
        self.setup_styles()
        self.create_widgets()
        
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        self.style.configure("Stat.TLabel", font=("Consolas", 10))

    def create_widgets(self):
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl, text="Container:").pack(side=tk.LEFT, padx=5)
        self.c_var = tk.StringVar(value="40GP")
        ttk.Combobox(ctrl, textvariable=self.c_var, values=list(CONTAINERS.keys()), width=10).pack(side=tk.LEFT, padx=5)

        ttk.Button(ctrl, text="Upload Excel", command=self.upload).pack(side=tk.LEFT, padx=10)
        ttk.Button(ctrl, text="Calculate Layout", command=self.run).pack(side=tk.LEFT)

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.tab1 = ttk.Frame(self.notebook); self.notebook.add(self.tab1, text="Isometric View")
        self.tab2 = ttk.Frame(self.notebook); self.notebook.add(self.tab2, text="Top View")

        self.side_panel = ttk.LabelFrame(main_frame, text=" Packing Summary ", padding=10)
        self.side_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=5)
        
        self.summary_text = tk.Text(self.side_panel, width=35, height=20, font=("Consolas", 10), state='disabled', bg="#f0f0f0")
        self.summary_text.pack(fill=tk.BOTH, expand=True)

    def upload(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if not path:
            return

        try:
            df = pd.read_excel(path)
            self.items_data = []

            for idx, r in df.iterrows():
                raw_sku = r.get('SKU', '')
                if pd.isna(raw_sku) or str(raw_sku).strip().lower() == 'nan' or str(raw_sku).strip() == '':
                    sku = f"Item_{idx+1}"
                else:
                    sku = str(raw_sku)
                
                length = float(r['Length'])
                width  = float(r['Width'])
                height = float(r['Height'])
                
                weight = float(r.get('Gross Wt / Plt', 0.0))
                if pd.isna(weight):
                    weight = 0.0
                
                self.items_data.append(Item(sku, length, width, height, weight))

            messagebox.showinfo("Ready", f"Loaded {len(self.items_data)} pallets.")
            self.update_summary(initial=True)
            
        except KeyError as e:
            messagebox.showerror("Error", f"Could not find column: {e}.\nPlease ensure your columns are exactly: SKU, Length, Width, Height, Gross Wt / Plt")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def update_summary(self, initial=False):
        self.summary_text.config(state='normal')
        self.summary_text.delete('1.0', tk.END)

        counts = Counter([i.sku for i in self.items_data])

        self.summary_text.insert(tk.END, "ITEMS LOADED:\n")
        self.summary_text.insert(tk.END, "-"*30 + "\n")
        
        display_counts = list(counts.items())[:10]
        for sku, count in display_counts:
            self.summary_text.insert(tk.END, f"SKU {sku}: {count} units\n")
        if len(counts) > 10:
            self.summary_text.insert(tk.END, f"...and {len(counts)-10} more SKUs\n")

        if self.packer:
            placed_counts = Counter([i.sku for i in self.packer.placed])
            failed_counts = Counter([i.sku for i in self.packer.unplaced])

            self.summary_text.insert(tk.END, "\n\nOPTIMIZATION RESULT\n")
            self.summary_text.insert(tk.END, "="*30 + "\n")
            self.summary_text.insert(tk.END, f"Container   : {self.c_var.get()}\n")
            self.summary_text.insert(tk.END, f"Total Items : {self.total_items}\n")
            self.summary_text.insert(tk.END, f"Placed      : {self.placed_items}\n")
            self.summary_text.insert(tk.END, f"Failed      : {self.failed_items}\n")
            self.summary_text.insert(tk.END, f"Utilization : {self.utilization:.2f}%\n")
            self.summary_text.insert(tk.END, "="*30 + "\n\n")

            self.summary_text.insert(tk.END, "SKU BREAKDOWN\n")
            self.summary_text.insert(tk.END, "-"*30 + "\n")

            all_skus = sorted(set(list(placed_counts.keys()) + list(failed_counts.keys())))

            for sku in all_skus:
                p = placed_counts.get(sku, 0)
                f = failed_counts.get(sku, 0)
                self.summary_text.insert(tk.END, f"SKU {sku} | ✔ {p} | ❌ {f}\n")
            
            if self.packer.unplaced:
                self.summary_text.insert(tk.END, "\n\nFAILED PALLETS (WON'T FIT)\n")
                self.summary_text.insert(tk.END, "-"*30 + "\n")
                for unplaced_item in self.packer.unplaced:
                    self.summary_text.insert(tk.END, f"❌ {unplaced_item.sku}\n")
                    self.summary_text.insert(tk.END, f"   Wt:{unplaced_item.weight}kg L:{unplaced_item.dims[0]} W:{unplaced_item.dims[1]} H:{unplaced_item.dims[2]}\n")

        self.summary_text.config(state='disabled')

    def run(self):
        if not self.items_data:
            messagebox.showwarning("Warning", "Please upload data first!")
            return

        c_info = CONTAINERS[self.c_var.get()]
        
        items_to_pack = [Item(i.sku, i.dims[0], i.dims[1], i.dims[2], i.weight) for i in self.items_data]
        
        self.packer = OptimizationEngine((c_info['L'], c_info['W'], c_info['H']))
        self.packer.pack(items_to_pack)

        total = len(items_to_pack)
        placed = len(self.packer.placed)
        failed = len(self.packer.unplaced)

        utilization = (sum(i.vol for i in self.packer.placed) / 
                    (c_info['L'] * c_info['W'] * c_info['H'])) * 100

        print("\n========== OPTIMIZATION RESULT ==========")
        print(f"Container: {self.c_var.get()}")
        print(f"Total Items: {total}")
        print(f"Placed: {placed}")
        print(f"Unplaced: {failed}")
        print(f"Utilization: {utilization:.2f}%")
        print("=========================================\n")
        
        self.total_items = total
        self.placed_items = placed
        self.failed_items = failed
        self.utilization = utilization

        self.draw_all_views(c_info)
        self.update_summary()

    def draw_all_views(self, c):
        for tab in [self.tab1, self.tab2]:
            for w in tab.winfo_children(): w.destroy()
        self.create_plot(self.tab1, c, 25, -60, "Isometric View")
        self.create_plot(self.tab2, c, 90, -90, "Floor Plan View")

    def create_plot(self, frame, c, elev, azim, title):
        fig = plt.Figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot3D([0, c['L'], c['L'], 0, 0], [0, 0, c['W'], c['W'], 0], [0, 0, 0, 0, 0], color='gray', alpha=0.5)
        
        skus = sorted(list(set(i.sku for i in self.packer.placed)))
        color_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i in self.packer.placed:
            color = color_palette[skus.index(i.sku) % len(color_palette)]
            
            ax.bar3d(i.pos[0], i.pos[1], i.pos[2], 
                     i.curr_dims[0], i.curr_dims[1], i.curr_dims[2], 
                     color=color, edgecolor='black', alpha=0.7, linewidth=0.3)

        ax.set_box_aspect((c['L'], c['W'], c['H']))
        ax.view_init(elev=elev, azim=azim)
        ax.set_zlim(0, c['H'])
        ax.set_title(title)
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        annot = ax.text2D(0.05, 0.90, "", transform=ax.transAxes, 
                          bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="gray", alpha=0.9),
                          fontsize=9)
        annot.set_visible(False)

        def on_hover(event):
            if event.inaxes == ax:
                closest_item = None
                min_dist = float('inf')
                
                for item in self.packer.placed:
                    cx = item.pos[0] + item.curr_dims[0] / 2
                    cy = item.pos[1] + item.curr_dims[1] / 2
                    cz = item.pos[2] + item.curr_dims[2] / 2
                    
                    x2, y2, _ = proj3d.proj_transform(cx, cy, cz, ax.get_proj())
                    display_coords = ax.transData.transform((x2, y2))
                    
                    dist = ((event.x - display_coords[0])**2 + (event.y - display_coords[1])**2)**0.5
                    
                    if dist < 40 and dist < min_dist:
                        min_dist = dist
                        closest_item = item
                        
                if closest_item:
                    text = (f"📦 SKU: {closest_item.sku} | ⚖️ Wt: {closest_item.weight:.1f} kg\n"
                            f"📐 Dims: L:{closest_item.curr_dims[0]:.1f} W:{closest_item.curr_dims[1]:.1f} H:{closest_item.curr_dims[2]:.1f} cm\n"
                            f"📍 Pos: X:{closest_item.pos[0]:.1f} Y:{closest_item.pos[1]:.1f} Z:{closest_item.pos[2]:.1f} cm")
                    annot.set_text(text)
                    annot.set_visible(True)
                    canvas.draw_idle()
                else:
                    if annot.get_visible():
                        annot.set_visible(False)
                        canvas.draw_idle()

        canvas.mpl_connect("motion_notify_event", on_hover)
        canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    ProfessionalApp(root)
    root.mainloop()

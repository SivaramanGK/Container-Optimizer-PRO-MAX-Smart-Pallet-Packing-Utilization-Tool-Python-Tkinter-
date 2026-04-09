import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import itertools
from collections import Counter

# --- Container Constants (mm) ---
CONTAINERS = {
    "20GP": {"L": 5898, "W": 2352, "H": 2393},
    "40GP": {"L": 12032, "W": 2352, "H": 2393},
    "40HC": {"L": 12032, "W": 2352, "H": 2698}
}

class Item:
    def __init__(self, sku, l, w, h):
        self.sku = sku
        self.dims = [l, w, h]
        self.vol = l * w * h
        self.pos = [0, 0, 0]
        self.curr_dims = [l, w, h]

class OptimizationEngine:
    def __init__(self, c_dims):
        self.L, self.W, self.H = c_dims
        self.placed = []
        self.unplaced = []

    def intersects(self, x, y, z, l, w, h):
        if x + l > self.L or y + w > self.W or z + h > self.H:
            return True
        for p in self.placed:
            px, py, pz = p.pos
            pl, pw, ph = p.curr_dims
            if not (x + l <= px or x >= px + pl or 
                    y + w <= py or y >= py + pw or 
                    z + h <= pz or z >= pz + ph):
                return True
        return False

    def pack(self, items):
        items.sort(key=lambda x: x.vol, reverse=True)
        for item in items:
            best_pt = None
            best_dims = None
            min_score = float('inf')

            pts = [[0, 0, 0]]
            for p in self.placed:
                pts.append([p.pos[0] + p.curr_dims[0], p.pos[1], p.pos[2]])
                pts.append([p.pos[0], p.pos[1] + p.curr_dims[1], p.pos[2]])
                pts.append([p.pos[0], p.pos[1], p.pos[2] + p.curr_dims[2]])

            pts.sort(key=lambda p: (p[2], p[0], p[1]))

            for pt in pts:
                # Pallet Logic: Only 2 horizontal rotations (keep height vertical)
                orientations = [(item.dims[0], item.dims[1], item.dims[2]),
                               (item.dims[1], item.dims[0], item.dims[2])]
                
                for dims in orientations:
                    if not self.intersects(pt[0], pt[1], pt[2], dims[0], dims[1], dims[2]):
                        score = pt[0] + pt[1] + pt[2]
                        if score < min_score:
                            min_score = score
                            best_pt, best_dims = pt, dims
                if best_pt: break

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
        # Top Control Bar
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl, text="Container:").pack(side=tk.LEFT, padx=5)
        self.c_var = tk.StringVar(value="40GP")
        ttk.Combobox(ctrl, textvariable=self.c_var, values=list(CONTAINERS.keys()), width=10).pack(side=tk.LEFT, padx=5)

        ttk.Button(ctrl, text="Upload Excel", command=self.upload).pack(side=tk.LEFT, padx=10)
        ttk.Button(ctrl, text="Calculate Layout", command=self.run).pack(side=tk.LEFT)

        # Main Content Area
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left side: 3D Visualization
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.tab1 = ttk.Frame(self.notebook); self.notebook.add(self.tab1, text="Isometric View")
        self.tab2 = ttk.Frame(self.notebook); self.notebook.add(self.tab2, text="Top View")

        # Right side: SKU Summary Panel
        self.side_panel = ttk.LabelFrame(main_frame, text=" Packing Summary ", padding=10)
        self.side_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=5)
        
        self.summary_text = tk.Text(self.side_panel, width=30, height=20, font=("Consolas", 10), state='disabled', bg="#f0f0f0")
        self.summary_text.pack(fill=tk.BOTH, expand=True)

    def upload(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if not path:
            return

        df = pd.read_excel(path)

        self.items_data = []

        for idx, r in df.iterrows():
            # Use row number as SKU (since no SKU column)
            sku = f"Pallet_{idx+1}"

            length = float(r['Length']) * 10   # cm → mm
            width  = float(r['Width']) * 10
            height = float(r['Height']) * 10

            self.items_data.append(Item(sku, length, width, height))

        messagebox.showinfo("Ready", f"Loaded {len(self.items_data)} pallets.")

        self.update_summary(initial=True)
    def update_summary(self, initial=False):
        self.summary_text.config(state='normal')
        self.summary_text.delete('1.0', tk.END)

        # ✅ ALWAYS show loaded items
        counts = Counter([i.sku for i in self.items_data])

        self.summary_text.insert(tk.END, "ITEMS LOADED:\n")
        self.summary_text.insert(tk.END, "-"*30 + "\n")
        for sku, count in sorted(counts.items()):
            self.summary_text.insert(tk.END, f"SKU {sku}: {count} units\n")

        # ✅ If optimization done → show results ALSO
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
                self.summary_text.insert(
                    tk.END,
                    f"SKU {sku}\n  ✔ Placed : {p}\n  ❌ Failed : {f}\n" + "-"*20 + "\n"
                )

        self.summary_text.config(state='disabled')
    def run(self):
            if not self.items_data:
                messagebox.showwarning("Warning", "Please upload data first!")
                return

            c_info = CONTAINERS[self.c_var.get()]
            self.packer = OptimizationEngine((c_info['L'], c_info['W'], c_info['H']))

            # Fresh copy
            items_to_pack = [Item(i.sku, i.dims[0], i.dims[1], i.dims[2]) for i in self.items_data]
            self.packer.pack(items_to_pack)

            total = len(items_to_pack)
            placed = len(self.packer.placed)
            failed = len(self.packer.unplaced)

            utilization = (sum(i.vol for i in self.packer.placed) / 
                        (c_info['L'] * c_info['W'] * c_info['H'])) * 100

            # 👉 PRINT IN TERMINAL
            print("\n========== OPTIMIZATION RESULT ==========")
            print(f"Container: {self.c_var.get()}")
            print(f"Total Items: {total}")
            print(f"Placed: {placed}")
            print(f"Unplaced: {failed}")
            print(f"Utilization: {utilization:.2f}%")
            print("=========================================\n")
            # Draw charts + summary
            # ✅ STORE FIRST
            self.total_items = total
            self.placed_items = placed
            self.failed_items = failed
            self.utilization = utilization

            # ✅ THEN UPDATE UI
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
        
        # Container Outline
        ax.plot3D([0, c['L'], c['L'], 0, 0], [0, 0, c['W'], c['W'], 0], [0, 0, 0, 0, 0], color='gray', alpha=0.5)
        
        skus = sorted(list(set(i.sku for i in self.packer.placed)))
        # Updated Color Logic to avoid Deprecation Warning
        cmap = mpl.colormaps['tab10'].resampled(len(skus)) if skus else None

        for i in self.packer.placed:
            color = cmap(skus.index(i.sku)) if cmap else 'blue'
            ax.bar3d(i.pos[0], i.pos[1], i.pos[2], 
                     i.curr_dims[0], i.curr_dims[1], i.curr_dims[2], 
                     color=color, edgecolor='black', alpha=0.7, linewidth=0.3)

        ax.set_box_aspect((c['L'], c['W'], c['H']))
        ax.view_init(elev=elev, azim=azim)
        ax.set_zlim(0, c['H']) # Critical for seeing height differences
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    ProfessionalApp(root)
    root.mainloop()

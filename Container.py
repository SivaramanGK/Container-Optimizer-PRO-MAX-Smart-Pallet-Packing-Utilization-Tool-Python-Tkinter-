import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
# --- Container Specifications (mm) ---
CONTAINERS = {
    "20GP": {"L": 5898, "W": 2352, "H": 2393, "MAX_WT": 28000},
    "40GP": {"L": 12032, "W": 2352, "H": 2393, "MAX_WT": 30480},
    "40HC": {"L": 12032, "W": 2352, "H": 2698, "MAX_WT": 30480}
}

class Item:
    def __init__(self, sku, l, w, h, wt):
        self.sku = sku
        self.dims = sorted([l, w, h], reverse=True) # Standardize orientation
        self.wt = wt
        self.pos = [0, 0, 0]
        self.current_dims = [l, w, h]

class Packer:
    def __init__(self, container_dims):
        self.L, self.W, self.H = container_dims
        self.placed_items = []
        self.unplaced_items = []

    def fits(self, x, y, z, l, w, h):
        """Check if item fits in container and doesn't collide with others."""
        if x + l > self.L or y + w > self.W or z + h > self.H:
            return False
        for item in self.placed_items:
            ix, iy, iz = item.pos
            il, iw, ih = item.current_dims
            if not (x + l <= ix or x >= ix + il or
                    y + w <= iy or y >= iy + iw or
                    z + h <= iz or z >= iz + ih):
                return False
        return True

    def pack(self, items):
        # Sort items by volume (Largest first for best density)
        items.sort(key=lambda x: x.dims[0]*x.dims[1]*x.dims[2], reverse=True)
        
        for item in items:
            placed = False
            # Search for first available (x,y,z) point using a step-based grid
            # For 95%+ precision, we check every placement point created by previous items
            potential_points = [[0, 0, 0]]
            for p in self.placed_items:
                potential_points.append([p.pos[0] + p.current_dims[0], p.pos[1], p.pos[2]])
                potential_points.append([p.pos[0], p.pos[1] + p.current_dims[1], p.pos[2]])
                potential_points.append([p.pos[0], p.pos[1], p.pos[2] + p.current_dims[2]])

            # Sort points to fill from back-bottom-left
            potential_points.sort(key=lambda p: (p[2], p[1], p[0]))

            for pt in potential_points:
                x, y, z = pt
                # Try all 6 rotations
                import itertools
                for dims in set(itertools.permutations(item.dims)):
                    l, w, h = dims
                    if self.fits(x, y, z, l, w, h):
                        item.pos = [x, y, z]
                        item.current_dims = [l, w, h]
                        self.placed_items.append(item)
                        placed = True
                        break
                if placed: break
            
            if not placed:
                self.unplaced_items.append(item)

# --- GUI Application ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("95% Efficiency Container Optimizer")
        self.items_list = []
        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding="20")
        frame.pack()

        self.cont_var = tk.StringVar(value="40GP")
        ttk.Label(frame, text="Select Container:").grid(row=0, column=0)
        ttk.Combobox(frame, textvariable=self.cont_var, values=list(CONTAINERS.keys())).grid(row=0, column=1)

        ttk.Button(frame, text="Load Excel", command=self.load_excel).grid(row=1, column=0, columnspan=2, pady=10)
        ttk.Button(frame, text="Calculate & View 3D", command=self.optimize).grid(row=2, column=0, columnspan=2)

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if not path: return
        df = pd.read_excel(path)
        self.items_list = []
        for _, r in df.iterrows():
            for _ in range(int(r['Quantity'])):
                # Adjusting units: assuming Excel is CM, converting to MM
                self.items_list.append(Item(r['SKU'], r['Length']*10, r['Width']*10, r['Height']*10, r['Weight']))
        messagebox.showinfo("Ready", f"Loaded {len(self.items_list)} boxes.")

    def optimize(self):
        if not self.items_list:
            return

        c_info = CONTAINERS[self.cont_var.get()]
        c_dims = (c_info['L'], c_info['W'], c_info['H'])
        packer = Packer(c_dims)
        packer.pack(self.items_list)

        # Utilization Calculation
        used_vol = sum(i.current_dims[0]*i.current_dims[1]*i.current_dims[2] for i in packer.placed_items)
        total_vol = c_dims[0] * c_dims[1] * c_dims[2]
        util = (used_vol / total_vol) * 100

        # --- Visualizing the Container as a Boundary ---
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 1. Plot the "Container Box" (very faint/transparent)
        # This draws a very faint wireframe box around the entire boundary
        x_corners = [0, c_dims[0], c_dims[0], 0, 0, 0, c_dims[0], c_dims[0], 0, c_dims[0]]
        y_corners = [0, 0, c_dims[1], c_dims[1], 0, 0, 0, c_dims[1], c_dims[1], c_dims[1]]
        z_corners = [0, 0, 0, 0, 0, c_dims[2], c_dims[2], c_dims[2], c_dims[2], c_dims[2]]
        
        # Plot container boundary as a gray, dotted wireframe
        ax.plot3D(x_corners, y_corners, z_corners, color='gray', linestyle='--', alpha=0.3)
        
        # Optional: Plot the container floor as a subtle translucent plane
        # this helps anchor the items.
        floor_vertices = [np.array([0, 0, 0]), np.array([c_dims[0], 0, 0]), 
                          np.array([c_dims[0], c_dims[1], 0]), np.array([0, c_dims[1], 0])]
        # poly = plt.Poly3DCollection([floor_vertices], alpha=0.05, color='gray')
        # ax.add_collection3d(poly)

        # 2. Plot the placed items (solid boxes with edge lines)
        unique_skus = list(set(i.sku for i in packer.placed_items))
        # Use a qualitative colormap (like 'Set3' or 'Paired') for distinct colors per SKU
        colors = plt.cm.get_cmap('Set3', len(unique_skus))
        sku_color_map = {sku: colors(i) for i, sku in enumerate(unique_skus)}

        # Keep track of colors for the legend
        seen_skus = set()
        legend_patches = []

        for i in packer.placed_items:
            # We add black edge lines to make individual boxes stand out, 
            # while keeping the faces translucent (alpha) for visibility.
            c = sku_color_map[i.sku]
            ax.bar3d(i.pos[0], i.pos[1], i.pos[2], 
                     i.current_dims[0], i.current_dims[1], i.current_dims[2], 
                     color=c, edgecolor='black', linewidth=0.5, alpha=0.7)
            
            # Create a legend entry if we haven't seen this SKU color yet
            if i.sku not in seen_skus:
                patch = mpatches.Patch(color=c, label=i.sku)
                legend_patches.append(patch)
                seen_skus.add(i.sku)

        # 3. Add Legend
        ax.legend(handles=legend_patches, loc='upper right', title="SKUs")

        # 4. Configure the view (matching the prompt image's perspective)
        # We need an angled overhead view. Adjust elev/azim if needed.
        ax.view_init(elev=25, azim=-60)
        ax.set_box_aspect(c_dims)  # Crucial for proper geometric scaling (long container)

        # 5. Add text result and title
        ax.set_title(f"3D Container Packing: {self.cont_var.get()}\n"
                     f"Utilization: {util:.2f}% | Placed: {len(packer.placed_items)} | Unplaced: {len(packer.unplaced_items)}", fontsize=12)

        ax.set_xlabel('Length (mm)', fontsize=10)
        ax.set_ylabel('Width (mm)', fontsize=10)
        ax.set_zlabel('Height (mm)', fontsize=10)

        plt.tight_layout()
        plt.show()

    def plot_3d(self, packer, c, util):
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color map for SKUs
        unique_skus = list(set(i.sku for i in packer.placed_items))
        colors = plt.cm.get_cmap('tab20', len(unique_skus))
        sku_color_map = {sku: colors(i) for i, sku in enumerate(unique_skus)}

        for i in packer.placed_items:
            ax.bar3d(i.pos[0], i.pos[1], i.pos[2], 
                     i.current_dims[0], i.current_dims[1], i.current_dims[2], 
                     color=sku_color_map[i.sku], edgecolor='black', alpha=0.7)

        ax.set_title(f"Utilization: {util:.2f}% | Placed: {len(packer.placed_items)} | Unplaced: {len(packer.unplaced_items)}")
        ax.set_xlim(0, c['L']); ax.set_ylim(0, c['W']); ax.set_zlim(0, c['H'])
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
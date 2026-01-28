import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def apply_dark_theme(root):
    style = ttk.Style(root)
    style.theme_use("clam")

    bg = "#1e1e1e"
    fg = "#ffffff"
    accent = "#3a3a3a"

    root.configure(bg=bg)

    style.configure("TFrame", background=bg)
    style.configure("TLabelFrame", background=bg, foreground=fg, bordercolor=accent)
    style.configure("TLabel", background=bg, foreground=fg)
    style.configure("TButton",
                    background=accent,
                    foreground=fg,
                    padding=8,
                    relief="flat",
                    font=("Segoe UI", 11))
    style.map("TButton",
              background=[("active", "#505050")])


root = tk.Tk()
root.title("Neural Network Visualizer")
root.geometry("1200x700")

apply_dark_theme(root)

# ---------------- HEADER ----------------
header = ttk.Frame(root, padding=15)
header.grid(row=0, column=0, columnspan=3, sticky="ew")

title_label = ttk.Label(header, text="Neural Network Visualizer", font=("Segoe UI Semibold", 20))
title_label.pack(anchor="center")

# ---------------- BUTTON ----------------
button_frame = ttk.Frame(root)
button_frame.grid(row=1, column=2, sticky="w", padx=15, pady=10)

btn_start = ttk.Button(button_frame, text="Start Simulation")
btn_start.pack(anchor="w")

# ---------------- CANVAS ----------------
canvas_node = tk.Canvas(root, bg="#252526", highlightthickness=0)
canvas_node.grid(row=2, column=0, columnspan=3, padx=15, pady=10, sticky="nsew")

canvas_node.create_text(250, 20, text="Node Layers", fill="white", font=("Segoe UI", 12))

# ---------------- BOTTOM FRAMES ----------------
frame_numbers = ttk.LabelFrame(root, text="Numbers", padding=10)
frame_numbers.grid(row=3, column=0, padx=15, pady=10, sticky="nsew")

frame_accuracy = ttk.LabelFrame(root, text="Accuracy vs Iterations", padding=10)
frame_accuracy.grid(row=3, column=1, padx=15, pady=10, sticky="nsew")

frame_cost = ttk.LabelFrame(root, text="Cost vs Iterations", padding=10)
frame_cost.grid(row=3, column=2, padx=15, pady=10, sticky="nsew")

# Make bottom row taller
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=3)

# Make bottom frames expand
for f in (frame_numbers, frame_accuracy, frame_cost):
    f.grid_propagate(False)
    f.configure(height=250)


# ---------------- GRID WEIGHTS ----------------
# Canvas expands vertically
root.grid_rowconfigure(2, weight=3)
root.grid_rowconfigure(3, weight=2)

# Bottom frame width ratio 1 : 2 : 2
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=2)
root.grid_columnconfigure(2, weight=2)

root.mainloop()

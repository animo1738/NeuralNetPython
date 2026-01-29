import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from machinelearning import execute_llm 
import threading
import numpy as np
from sklearn.datasets import fetch_openml
import time 
# At the top of your script, under imports
print("Loading MNIST data...")
mnist = fetch_openml(name="mnist_784", version=1, as_frame=False)
mnist_data = mnist.data 
print("Data loaded.")
is_simulation = False
# global boolean flag to check status of simulation running 
def run_simulation_wrapper():
    global is_simulation
    is_simulation = True
    try:
        execute_llm(log_to_terminal)
    except Exception as e:
        log_to_terminal(f"Error: {e}")
    finally:
        is_simulation = False
        log_to_terminal("Simulation Finished.")

def start_simulation_thread():
    # We target the wrapper, not the raw execute_llm
    sim_thread = threading.Thread(target=run_simulation_wrapper, daemon=True)
    sim_thread.start()
    
def cycle_images():
    if is_simulation:
      
        n = np.random.randint(0, len(mnist_data))
        img_data = mnist_data[n]
        import_mnist_image(img_data, image_display)
        
    # Check again in 1000ms (1 second), regardless of the flag
    # This keeps the "loop" alive waiting for the simulation to start
    root.after(1000, cycle_images)        

def import_mnist_image(image_array, label_widget):
    reshaped = image_array.reshape(28, 28)
    img = Image.fromarray((reshaped * 255).astype('uint8'))
    img = img.resize((200, 200), Image.Resampling.NEAREST)
    tk_img = ImageTk.PhotoImage(img)
    label_widget.config(image=tk_img)
    label_widget.image = tk_img

def random_mnist():
    mnist = fetch_openml(name="mnist_784")
    data = mnist.data
    labels = mnist.target
    n = np.random.randint(0, data.shape[0])
    #the random choice of the dataset

    test_img = data.iloc[n].values
    test_label = mnist.target.iloc[n]

    return data.iloc[n].values



def log_to_terminal(message):
    canvas_node.insert(tk.END, f"> {message}\n")
    canvas_node.see(tk.END)      

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
button_frame.grid(row=1, column=0, sticky="w", padx=15, pady=10)


btn_start = ttk.Button(button_frame, text="Start Simulation", command = start_simulation_thread)
btn_start.pack(anchor="w")

# ---------------- CANVAS ----------------
canvas_node = tk.Text(root, bg="#1e1e1e", fg="#00ff00", 
                          insertbackground="white", font=("Consolas", 10),
                          height=10)
canvas_node.grid(row=3, column=0, columnspan=3, padx=15, pady=10, sticky="nsew")



# ---------------- BOTTOM FRAMES ----------------
frame_numbers = ttk.LabelFrame(root, text="Numbers", padding=10)
frame_numbers.grid(row=2, column=0, padx=15, pady=10, sticky="nsew")
image_display = ttk.Label(frame_numbers)
image_display.pack(pady=10)

cycle_images()


frame_accuracy = ttk.LabelFrame(root, text="Accuracy vs Iterations", padding=10)
frame_accuracy.grid(row=2, column=1, padx=15, pady=10, sticky="nsew")

frame_cost = ttk.LabelFrame(root, text="Cost vs Iterations", padding=10)
frame_cost.grid(row=2, column=2, padx=15, pady=10, sticky="nsew")

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
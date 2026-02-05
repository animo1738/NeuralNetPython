import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from machinelearning import execute_llm
import threading
import numpy as np
from sklearn.datasets import fetch_openml
import time 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

#---------GLOBAL VARIABLES--------------#
trained_network = None
fig_acc = Figure(figsize=(5, 4), dpi=100)
fig_cost = Figure(figsize=(5, 4), dpi=100)
canvas_cost = None
canvas_acc = None
AdjustedLR = None
AdjustedEpochs = None
mnist_data = None



is_simulation = False
# global boolean flag to check status of simulation running 
def run_simulation_wrapper():
    global is_simulation
    global trained_network
    global mnist_data
    selected_epochs = epoch_slider.get()
    selected_lr = learningrate_slider.get()
    try:
        log_to_terminal("Loading MNIST data...")
        mnist = fetch_openml(name="mnist_784", version=1, as_frame=False)
        mnist_data = mnist.data 
        log_to_terminal("Data loaded.")
        trained_network = execute_llm(log_to_terminal, update_border,live_update, selected_epochs, selected_lr)
    except Exception as e:
        log_to_terminal(f"Error: {e}")
    finally:
        is_simulation = False
        log_to_terminal("Simulation Finished.")
        root.after(0, lambda: btn_reset.config(state="normal"))
        root.after(0, lambda: btn_start.config(state="normal"))

        

def start_simulation_thread():
    # We target the wrapper, not the raw execute_llm
    
    sim_thread = threading.Thread(target=run_simulation_wrapper, daemon=True)
    btn_start.config(state="disabled")
    btn_reset.config(state="disabled")
    
    sim_thread.start()

def live_update(stats):
    global trained_network
    global is_simulation
    trained_network = stats.get('nn_instance')
    is_simulation = True
    message = f"Epoch {stats['epoch']}: Loss {stats['cost']:.4f} | Acc {stats['train_acc']:.2f}%"
    root.after(0, lambda: log_to_terminal(message))
    root.after(0, lambda: update_border(stats['train_acc']))
    root.after(0,lambda: cycle_images())
    if stats['epoch'] % 5 == 0:
        root.after(0, plot_graphs)

def cycle_images():
    global mnist_data
    if is_simulation:
      
        n = np.random.randint(0, len(mnist_data))
        img_data = mnist_data[n]
        import_mnist_image(img_data, image_display)
        
    # Check again in 1000ms (1 second), regardless of the flag
    # This keeps the "loop" alive waiting for the simulation to start
    root.after(1500, cycle_images)        

def import_mnist_image(image_array, label_widget):
    reshaped = image_array.reshape(28, 28)
    img = Image.fromarray((reshaped * 255).astype('uint8'))
    img = img.resize((95, 95), Image.Resampling.NEAREST)
    tk_img = ImageTk.PhotoImage(img)
    label_widget.config(image=tk_img)
    label_widget.image = tk_img



def plot_graphs():
    global trained_network, fig_acc, fig_cost, canvas_acc, canvas_cost
    if trained_network is None: return
    fig_acc.clear() 
    fig_cost.clear()
    plt.style.use('dark_background') 
    for f in [fig_acc, fig_cost]:
        f.set_facecolor('#1e1e1e')
    
    ax = fig_acc.add_subplot(111)
    ax.plot(trained_network.accuracies['train'], label="Train")
    ax.plot(trained_network.accuracies['test'], label="Test")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()

    ax2 = fig_cost.add_subplot(111)
    ax2.plot(trained_network.costs, color='red')
    ax2.set_title("Loss (Cost) over Epochs")
    

    # draw the canvas
    canvas_acc.draw()
    canvas_cost.draw()

def random_mnist():
    mnist = fetch_openml(name="mnist_784")
    data = mnist.data
    labels = mnist.target
    n = np.random.randint(0, data.shape[0])
    #the random choice of the dataset

    test_img = data.iloc[n].values
    test_label = mnist.target.iloc[n]
    return data.iloc[n].values 

def reset_simulation():
    global trained_network, is_simulation

    is_simulation = False
    trained_network = None

    canvas_node.delete("1.0", tk.END)

    epoch_slider.set(100)
    learningrate_slider.set(0.01)

    image_display.config(image="")
    image_display.image = None

    image_border_frame.config(bg="#ff0000")

    fig_acc.clear()
    fig_cost.clear()
    canvas_acc.draw()
    canvas_cost.draw()
    
    log_to_terminal("Simulation reset.")


def update_border(accuracy):
    import random
    colors = ['#00ff00','#ff0000' ]
    probabilities = [accuracy,100-accuracy ]
    color = random.choices(colors, weights=probabilities, k=1)[0]
    image_border_frame.config(bg=color)


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

# ---------------- BUTTON AND SCROLLBARS----------------

sidebar = ttk.Frame(root, padding=10)
sidebar.grid(row=1, column=0, rowspan=3, sticky="n", padx=15)

# Start Button
btn_start = ttk.Button(sidebar, text="Start Simulation", command=start_simulation_thread )
btn_start.pack(fill="x", pady=(0, 50))

btn_reset = ttk.Button(sidebar, text="Reset", command=reset_simulation)
btn_reset.pack(fill="x", pady=(0, 20))


# Epoch Slider
epoch_slider = tk.Scale(sidebar, from_=100, to=1000, orient="horizontal", 
    label="Amount of Epochs",
    bg="#1e1e1e", 
    fg="white", 
    highlightthickness=0,
    
)
epoch_slider.set(100)
epoch_slider.pack(fill="x", pady=10)

# Learning Rate Slider
learningrate_slider = tk.Scale(sidebar, from_= 0.01, to=0.05, resolution=0.01, 
    orient="horizontal", 
    label="Learning Rate",
    bg="#1e1e1e", 
    fg="white", 
    highlightthickness=0,
   
)
learningrate_slider.set(0.01)
learningrate_slider.pack(fill="x", pady=10)




# ---------------- CANVAS ----------------
canvas_node = tk.Text(root, bg="#1e1e1e", fg="#00ff00", 
                          insertbackground="white", font=("Consolas", 10),
                          height=10)
canvas_node.grid(row=3, column=1, columnspan=2, padx=15, pady=10, sticky="nsew")

# ---------------- BOTTOM FRAMES ----------------
frame_numbers = ttk.LabelFrame(root, text="Numbers", padding=10)
frame_numbers.grid(row=3, column=0, padx=15, pady=10, sticky="nsew")
image_border_frame = tk.Frame(frame_numbers, bg="#ff0000", padx=3, pady=3)
image_border_frame.pack(pady=10)
image_display = tk.Label(image_border_frame, bg="#1e1e1e")
image_display.pack()

frame_accuracy = ttk.LabelFrame(root, text="Accuracy vs Iterations", padding=10)
frame_accuracy.grid(row=2, column=1, padx=15, pady=10, sticky="nsew")
canvas_acc = FigureCanvasTkAgg(fig_acc, master = frame_accuracy)
toolbar_acc = NavigationToolbar2Tk(canvas_acc, frame_accuracy) 
toolbar_acc.update()
toolbar_acc.pack() 
canvas_acc.get_tk_widget().pack(fill=tk.BOTH, expand=True) 

frame_cost = ttk.LabelFrame(root, text="Cost vs Iterations", padding=10)
frame_cost.grid(row=2, column=2, padx=15, pady=10, sticky="nsew")
canvas_cost = FigureCanvasTkAgg(fig_cost, master = frame_cost)  
toolbar_cost = NavigationToolbar2Tk(canvas_cost, frame_cost) 
toolbar_cost.update()
toolbar_cost.pack()
canvas_cost.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Make bottom row taller
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=3)

# Make bottom frames expand
for f in (frame_numbers, frame_accuracy, frame_cost):
    f.grid_propagate(True)
    f.configure(height=300)


# ---------------- GRID WEIGHTS ----------------
# Canvas expands vertically
root.grid_rowconfigure(0, weight=0)
root.grid_rowconfigure(1, weight = 0)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=0)
root.grid_rowconfigure(3, minsize=200)
# Bottom frame width ratio 1 : 2 : 2
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=2)
root.grid_columnconfigure(2, weight=2)
root.grid_columnconfigure(0, minsize=200)
root.grid_columnconfigure(1, minsize=440)
root.grid_columnconfigure(2, minsize=440)
root.mainloop()

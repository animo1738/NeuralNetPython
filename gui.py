import tkinter as tk
from sklearn.datasets import fetch_openml
import numpy as np
from PIL import Image, ImageTk

#importing the mnist database as image to display.
mnist = fetch_openml(name="mnist_784")
data = mnist.data.to_numpy()

current_index = 0
running = False


# def load_image(index):
#     img_array = data[index].reshape(28,28).astype(np.uint8)
#     pil_img = Image.fromarray(img_array, mode="L").resize((200, 200))
#     return ImageTk.PhotoImage(pil_img)

# def update_image():
#     global current_index,tk_img
#     tk_img = load_image(current_index)
#     panel.config(image=tk_img)


# def auto_rotate():
#     global current_index, running 
#     if not running: 
#         return    
#     current_index = (current_index + 1) % len(data)
#     update_image()
#     root.after(200,auto_rotate)
# def start_rotation():
#     global running
#     running = True
#     auto_rotate() 
# def stop_rotation(): 
#     global running 
#     running = False

# configure window

root = tk.Tk()
# tk_img = load_image(current_index)

root.grid_rowconfigure(2, weight=1) 
root.grid_columnconfigure(3, weight=1)

root.loadimage = tk.PhotoImage(file="button_start-simulation.png")

btn_start = tk.Button(root, image=root.loadimage)
btn_start.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
btn_start["border"] = "0"

canvas_node = tk.Canvas(root, bg="white", width=500, height=300)
canvas_node.grid(row=1, column = 0, columnspan=4, padx=10, pady=10, sticky="nsew")

frame_numbers = tk.LabelFrame(root, text="Numbers", bg="#222", fg="white", font=("Segoe UI", 10))
frame_numbers.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")

frame_accuracy = tk.LabelFrame(root, text="Accuracy vs Iterations", bg="#222", fg="white", font=("Segoe UI", 10))
frame_accuracy.grid(row=2, column=1, padx=10, pady=5, sticky="nsew")

frame_cost = tk.LabelFrame(root, text="Cost vs Iterations", bg="#222", fg="white", font=("Segoe UI", 10))
frame_cost.grid(row=2, column=3, padx=10, pady=5, sticky="nsew")

root.grid_rowconfigure(2, weight=1) 
root.grid_columnconfigure(0, weight=1) 
root.grid_columnconfigure(1, weight=2)


# panel = tk.Label(root, image = tk_img)
# panel.pack()

# btn_start = tk.Button(root, text="Start Simulation", command=start_rotation)
# btn_start.pack()
# btn_stop = tk.Button(root, text="Stop", command=stop_rotation)
# btn_stop.pack()





root.mainloop()
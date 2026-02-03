# NeuralNetPython

A custom implementation of a **Neural Network Classifier**  built from scratch using NumPy. This project classifies handwritten digits from the MNIST dataset and includes a live monitoring GUI for visualizing the training process.

---

## Core Engine (`machinelearning.py`)

The heart of the project is a flexible Neural Network class that supports custom architectures and multiple activation functions.

### Key Features
* **Activation Functions**: Includes `sigmoid`, `relu`, `tanh`, and `leaky_relu`.
* **Layer Flexibility**: Dynamically initializes parameters based on a provided `architecture` list (e.g., hidden layer sizes).
* **Softmax Output**: Uses a `softmax` function at the final layer to produce a probability distribution for digit classification.
* **Backpropagation**: Manual implementation of gradient descent using the chain rule to update weights and biases.
* **Data Utilities**: Built-in functions for `normalise` and `one_hot_encode` to prepare MNIST data.

---

## Live Monitoring GUI

The project includes a **NeuralNet Live Monitor** built with PySide6 and Matplotlib to visualize model performance in real-time.

### `test_gui.py`
A standalone monitor providing:
* **MNIST Stream**: A panel that flashes random images from the dataset during the session.
* **Loss Graph**: A Matplotlib plot that procedurally grows to show training progress.
* **Dark Mode UI**: A professional grey-scale theme designed for high readability.

### `ui_visualiser.py`
A structured UI layout generated for dual-graph monitoring (Cost and Accuracy) with interactive controls like a "Start Simulation" button.

---

##  Requirements

To run this project, you will need the following Python libraries:
* `numpy` (Matrix operations)
* `PySide6` (GUI framework)
* `matplotlib` (Live plotting)
* `scikit-learn` (Fetching MNIST dataset)
* `tqdm` (Terminal progress bars)

---

## Usage

1. **Model Training**: The `NN` class in `machinelearning.py` can be instantiated with a custom architecture, such as `[128, 64]`, and trained using the `.fit()` method.
2. **GUI Visualization**: Running `test_gui.py` launches the "NeuralNet Live Monitor" window to observe training behavior visually.

---

## Credits
* **Author**: animo1738
* **Reference**: Based on the "Building a Neural Network from Scratch" tutorial by Bernardino Sassoli.
* **Date**: 18/01/2026

---


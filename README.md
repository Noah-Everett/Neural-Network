# **Neural Network**

This is the repository for the neural network I'm building from scratch in python using no external libraries. The propose of this project is not to make an efficient neural network that could perform tasks in a reasonable amount of time. Instead, I'm building this network to improve my understanding of python and neural networks, which is why I decided not to use any external libraries like TensorFlow, PyTorch, or even NumPy.

## Brief Explanation
`main.py` and `main.ipynb` are the main files. Their purpose is to run code from `network.py` and `input.py` which are the two primary files. `network.py` consists of a `Network` class that has functions for forward and back propagation along with a function to save the network to a `.json` file (which in my tests has been `Protein_Powder_Containers_Neural_Network.json`). `input.py` is responsible for providing the network with inputs that are usable (a 1D list). The last major component of the project is the `Data Sets` file, which includes all the photos I am using to train the network.

## Current State
Neural Network is currently very incomplete. I believe everything except back propagation is done.

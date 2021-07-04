# PINN PDE solver implementation

A basic implementation of Physical Informed Neural Networks in tensorflow2 (ref https://www.sciencedirect.com/science/article/pii/S0021999118307125).

* solver.py contains the Solver class which can solve a PDE given boundary, collocation points and a function defining the PDE. The LRSchedule class can be used to control the training of the neural network by decaying the learning rate.

* burger.ipynb and poisson.ipynb give two examples of resolution : the burger equation is solved on $[0,1]\times[0,1]$, and the Poisson (Laplace) equation is solved on a 2D cross-section of a simple circular cable. Both solutions are compared to exact solutions. 
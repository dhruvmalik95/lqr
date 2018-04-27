import numpy as np
import matplotlib.pyplot as plt
import math
import control
import scipy as sc

"""
This file is used to find dependency between the number of rollouts
required to get epsilon convergence in cost.

To run this, look at the bottom line of this file - the function called
there is the main one.
"""


"""""""""""""""""""""""
"""""""""""""""""""""""
"""LQR Setup"""
"""""""""""""""""""""""
"""""""""""""""""""""""


"""Work with a Normal(0,1) distribution the initial state"""
mu = 1

A = 0.8
B = 0.4
Q = 3
R = 4

K = 4.0

#note the value of m is really 2*m beacuse of my structure
r = 0.005

def is_stable(A, B, K):
	"""Return boolean to test whether this is stable or not."""
	return abs(A - B*K) < 1

#with variance = 1
def compute_gradient(A, B, Q, R, K):
	"""Returns gradient of the function assuming Normal(0, 1) distribution for 
	initial state."""
	numerator = 2*(A*A*(-1*R)*K + A*B*(R*K*K - Q) + K*(B*B*Q + R))
	denominator = (-1*A*A + 2*A*B*K - B*B*K*K + 1)**2
	return numerator / denominator
	
#this is actual cost with expectation over initial state with variance = 1
def compute_actual_cost(A, B, Q, R, K):
	"""Returns the actual cost of using the policy K."""
	numerator = (Q + K*R*K)*1
	denominator = 1 - ((A-B*K)**2)
	return numerator/denominator

#with variance = 1
def compute_actual_cost_from_state(A, B, Q, R, K, initial_state):
	"""Returns the cost of running a policy K from a particular initial state."""
	numerator = (Q + K*R*K)*(initial_state*initial_state)
	denominator = 1 - ((A-B*K)**2)
	return numerator/denominator


def find_minimizer(A, B, Q, R):
	"""Returns a tuple of things from solving a Riccatti equation, the
	third entry in the tuple is the optimal policy."""
	return control.dare(A, B, Q, R)

K_star = float(find_minimizer(np.array(A), np.array(B), np.array(Q), np.array(R))[2])
print("K_star: " + str(K_star))

def get_stable_region(A, B):
	return (1-A)/(-1*B), (1+A)/B

def plot_landscape(A, B, Q, R, density):
	lower, upper = get_stable_region(A, B)
	print(lower)
	print(upper)
	K_list = []
	cost_list = []
	for i in range(1, density):
		current_K = lower + i*(upper - lower)/density
		K_list.append(current_K)
		cost_list.append(compute_actual_cost(A, B, Q, R, current_K))
	plt.plot(K_list, cost_list)
	plt.xlabel("K")
	plt.ylabel("Cost")
	plt.show()

plot_landscape(A, B, Q, R, 1000)

"""Observations"""
"""
1. Multiplying both Q and R by a constant factor multiplies the cost
by the same constant factor.

2. Multiplying both A and B by a constant factor < 1 makes the region of
stability larger, and the costs around the edges of stability blow up faster.

3. Multiplying both A and B by a constant factor > 1 makes the region of
stability smaller, and the cost plot around the edges of stability does not
blow up as fast.

4. The one common theme is that throughout the experiments, near the regions of
stability the cost function is massive, and then it drops and looks like an almost
flat bowl in a relatively large area around the optimum.
"""




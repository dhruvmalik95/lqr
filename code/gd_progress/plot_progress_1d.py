import numpy as np
import matplotlib.pyplot as plt
import math
import control
import scipy as sc

A = 0.8
B = 0.4
Q = 3
R = 4

K = 4.485

initial_state = 15

def is_stable(A, B, K):
	return (A - B*K) < 1

def compute_next_state(A, B, K, current_state):
	return (A - B*K)*current_state

def compute_current_cost(Q, R, K, current_state):
	current_control = K*current_state
	return current_state*Q*current_state + current_control*R*current_control

def compute_correlation(A, B, K, current_state):
	state_list = []
	for _ in range(0, l):
		state_list.append(current_state)
		current_state = compute_next_state(A, B, K, current_state)

	correlation = 0
	for state in state_list:
		correlation = correlation + state*state
	return correlation

#with variance = 1
def compute_gradient(A, B, Q, R, K):
	numerator = 2*(A*A*(-1*R)*K + A*B*(R*K*K - Q) + K*(B*B*Q + R))*initial_state*initial_state
	denominator = (-1*A*A + 2*A*B*K - B*B*K*K + 1)**2
	return numerator / denominator
	
#this is actual cost with expectation over initial state with variance = 1
def compute_actual_cost(A, B, Q, R, K):
	numerator = (Q + K*R*K)*initial_state*initial_state
	denominator = 1 - ((A-B*K)**2)
	return numerator/denominator

def find_minimizer(A, B, Q, R):
	return control.dare(A, B, Q, R)

def plot_landscape(logarithm):
	cost_list = []
	for i in np.arange(start=-0.499, stop=4.49, step=0.001):
		print(i)
		cost_list.append(compute_actual_cost(A, B, Q, R, i))
	if logarithm==False:
		plt.plot(np.arange(start=-0.499, stop=4.49, step=0.001), cost_list, linewidth=2.5)
		plt.xlabel('Value Of K')
		plt.ylabel('Cost of K')
		plt.title('Cost Landscape')
		plt.show()
	else:
		plt.plot(np.arange(start=-0.499, stop=4.49, step=0.001), np.log(cost_list), linewidth=2.5)
		plt.xlabel('Value Of K')
		plt.ylabel('Log Of Cost Of K')
		plt.title('Log Of Cost Landscape')
		plt.show()
	return cost_list

K_star = float(find_minimizer(np.array(A), np.array(B), np.array(Q), np.array(R))[2])
print("K_star: "+ str(K_star))
landscape_cost_list = plot_landscape(logarithm=False)
plot_landscape(logarithm=True)

"""Now I'm going to try and run gradient descent, using the "exact" gradient computation,
even though of course sigma_k is really being estimated"""

def gradient_step(current_K, step_size, gradient):
	next_K = current_K - step_size*gradient
	return next_K

def gradient_descent(initial_K, step_size, epsilon, max_iter, optimum):
	i = 0
	current_K = initial_K
	
	iteration_list = []
	K_list = []
	
	while i < max_iter and abs(compute_actual_cost(A, B, Q, R, current_K) - compute_actual_cost(A, B, Q, R, K_star)) > epsilon:
		iteration_list.append(i)
		K_list.append(current_K)
		
		gradient = compute_gradient(A, B, Q, R, current_K)
		
		print("Iteration: " + str(i))
		print("Current K: " + str(current_K))
		print("Gradient: " + str(gradient))
		print("________________________")

		current_K = gradient_step(current_K, step_size, gradient)
		
		if not is_stable(A, B, current_K):
			print("At iteration " + str(i) + " we are no longer stable.")
			i = max_iter + 1587
		
		i = i + 1
	return iteration_list, K_list

# iteration_list, K_list = gradient_descent(K, 0.002, 10**(-6), 500, K_star)
# plt.plot(iteration_list, K_list)
# plt.show()

"""OK! So what we see is that regular GD behaves exactly as we would expect it to"""

"""Now the idea is to estimate the gradient using Kakade's method, instead of computing it
The following code is to perform rollouts to do exactly this"""

iteration_list, K_list = gradient_descent(K, 10000/(compute_actual_cost(A, B, Q, R, K)**2), 10**(-3), 10000000000, K_star)

mod_K_list = K_list[0::100]
mod_K_list_cost = [compute_actual_cost(A, B, Q, R, K) for K in mod_K_list]
plt.plot(100*np.array(range(0, len(mod_K_list))), mod_K_list_cost, linewidth=2.5)
plt.xlabel('Iterations')
plt.ylabel('Cost Of K')
plt.title('Gradient Descent Progress')
plt.show()

plt.plot(100*np.array(range(0, len(mod_K_list))), np.log(mod_K_list_cost), linewidth=2.5)
plt.xlabel('Iterations')
plt.ylabel('Log Of Cost Of K')
plt.title('Gradient Descent Progress')
plt.show()

number_points = len(K_list)//20
K_list = K_list[0::number_points]
cost_list = [compute_actual_cost(A, B, Q, R, K) for K in K_list]

plt.plot(np.arange(start=-0.499, stop=4.49, step=0.001), landscape_cost_list, c='b', linewidth=2.5, zorder=1)
plt.scatter(K_list, cost_list, c='r', s=35, zorder=2)
plt.xlabel('Value Of K')
plt.ylabel('Cost Of K')
plt.title('Descent Down Hill')
plt.show()

plt.plot(np.arange(start=-0.499, stop=4.49, step=0.001), np.log(landscape_cost_list), c='b', linewidth=2.5, zorder=1)
plt.scatter(K_list, np.log(cost_list), c='r', s=35, zorder=2)
plt.xlabel('Value Of K')
plt.ylabel('Log Of Cost Of K')
plt.title('Descent Down Hill')
plt.show()

# plt.plot(iteration_list, [compute_actual_cost(A, B, Q, R, K) for K in K_list])
# plt.show()


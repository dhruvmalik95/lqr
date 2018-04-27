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

K = 2.0

#note the value of m is really 2*m beacuse of my structure
r = 0.005

def sample_initial_state():
	"""Returns an initial_state sampled from Normal(0, 1) distribution."""
	return np.random.normal(loc=0.0, scale=1.0)

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

def compute_estimated_gradient(A, B, Q, R, K, m):
	"""Returns the gradient after it has been estimated from m rollouts."""
	cost_list = []
	U_list = []

	gradient = 0
	for _ in range(0, m):
		current_state = sample_initial_state()
		K_hat_1 = K + r
		cost_1 = compute_actual_cost_from_state(A, B, Q, R, K_hat_1, current_state)
		K_hat_2 = K - r
		cost_2 = compute_actual_cost_from_state(A, B, Q, R, K_hat_2, current_state)
		grad_estimate = (cost_1*r + cost_2*(-1*r))*(1/(r*r))
		gradient = gradient + grad_estimate

	#divide by an extra 2 because effectively for each of the 
	#m rollouts, I'm doing 2 rollouts, one with K+r and one
	#with K-r
	return (1/(2*m))*gradient

def find_minimizer(A, B, Q, R):
	"""Returns a tuple of things from solving a Riccatti equation, the
	third entry in the tuple is the optimal policy."""
	return control.dare(A, B, Q, R)

K_star = float(find_minimizer(np.array(A), np.array(B), np.array(Q), np.array(R))[2])
print("K_star: " + str(K_star))
#print("R good? " + str((compute_actual_cost(A, B, Q, R, K_star + r) - compute_actual_cost(A, B, Q, R, K_star - r))/(2*r)))


"""""""""""""""""""""""
"""""""""""""""""""""""
"""Functions to compute logarithmically scaled epsilon and m lists."""
"""""""""""""""""""""""
"""""""""""""""""""""""

def compute_m_test_list(lower, upper, density):
	"""Returns a logarithmically scaled list with density # of values
	given upper and lower bounds, for the values of m to test."""
	difference = (np.log(upper) - np.log(lower)) / density
	print(difference)
	l = []
	for i in np.arange(np.log(lower), np.log(upper), difference):
		l.append(int(round(np.exp(i))))
	return l

def compute_epsilon_list(lower, upper, density):
	"""Returns a logarithmically scaled list with density # of values
	given upper and lower bounds, for the values of epsilon for which
	to determine the minimum m."""
	difference = -1*(np.log(lower) - np.log(upper)) / density
	print(difference)
	l = []
	for i in np.arange(np.log(lower), np.log(upper), difference):
		l.append(np.exp(i))
	return l


"""""""""""""""""""""""
"""""""""""""""""""""""
"""Gradient Descent"""
"""""""""""""""""""""""
"""""""""""""""""""""""

def gradient_step(current_K, step_size, gradient):
	"""Takes a single gradient step and returns the next iterate."""
	next_K = current_K - step_size*gradient
	return next_K

def gradient_descent(initial_K, step_size, epsilon, max_iter, optimum, m):
	"""Runs GD until epsilon convergence to optimal cost. Has a stopping
	condition of when we get a gradient estimate which points in the
	opposite direction of the true gradient. Another stopping condition
	is when we hit a region of non-stability."""
	i = 0
	current_K = initial_K
	iterates = [current_K]
		
	while i < max_iter and abs(compute_actual_cost(A, B, Q, R, current_K) - compute_actual_cost(A, B, Q, R, optimum)) > epsilon:
		
		print("Iteration: " + str(i))
		print("Current K: " + str(current_K))

		estimated_gradient = compute_estimated_gradient(A, B, Q, R, current_K, m)
		#estimated_gradient = compute_gradient(A, B, Q, R, current_K)
		print("Estimated Gradient: " + str(estimated_gradient))
		print("Actual Gradient: " + str(compute_gradient(A, B, Q, R, current_K)))
		print("Actual Cost: " + str(compute_actual_cost(A, B, Q, R, current_K)))
		print("_____________________")

		if (estimated_gradient < 0 and compute_gradient(A, B, Q, R, current_K) > 0) or (estimated_gradient > 0 and compute_gradient(A, B, Q, R, current_K) < 0):
			#print("Stopping Condition, we have opposite gradient.")
			i = max_iter + 1690
		
		current_K = gradient_step(current_K, step_size, estimated_gradient)
		iterates.append(current_K)
		
		if not is_stable(A, B, current_K):
			#print("At iteration " + str(i) + " we are no longer stable.")
			i = max_iter + 1587
		
		i = i + 1

	# print("Final K: " + str(current_K))
	# print("Final K Cost: " + str(compute_actual_cost(A, B, Q, R, current_K)))
	# print("K_star Cost: " + str(compute_actual_cost(A, B, Q, R, K_star)))
	
	# print("Iterates: " + str(np.array(iterates)))
	# plt.plot(range(0, len(iterates)), iterates)
	# plt.xlabel("Iteration")
	# plt.ylabel("Current K")
	# plt.show()
	
	return current_K
	#return iterates

"""""""""""""""""""""""
"""""""""""""""""""""""
"""Functions to find m dependence on epsilon"""
"""""""""""""""""""""""
"""""""""""""""""""""""

def iterate_over_epsilon(epsilon_list, m_test_list):
	"""Iterates over input list of epsilons and returns a list
	containing the minimum value of m required to get epsilon
	convergence in cost for each epsilon."""
	m_list = []
	for epsilon in epsilon_list:
		print("current epsilon: " + str(epsilon))
		m = find_minimum_m(epsilon, m_test_list, max_iter)
		m_list.append(m)
	return m_list

def find_minimum_m(epsilon, m_test_list, max_iter):
	"""Given a fixed value of epsilon, this returns the minimum
	value of m required to get epsilon convergence in cost with
	probability at least delta."""
	for m in m_test_list:
		print("current m: " + str(m))
		if hypothesis_test(K, epsilon, m, num_simulations, max_iter) == True:
			return m
	print("m range too small")
	return 0

def hypothesis_test(K, epsilon, m, num_simulations, max_iter):
	"""Given a fixed value of epsilon and fixed value of m to test,
	this function returns a boolean of whether this value of m ensures
	epsilon convergence, with proportion at least delta, out of all the
	num_simulations it runs."""
	num_correct = 0
	for _ in range(0, num_simulations):
		estimated_optimum = gradient_descent(K, step_size, epsilon, max_iter, K_star, m)
		if abs(compute_actual_cost(A, B, Q, R, estimated_optimum) - compute_actual_cost(A, B, Q, R, K_star)) < epsilon:
			num_correct = num_correct + 1
	return (num_correct/num_simulations) > delta

"""""""""""""""""""""""
"""""""""""""""""""""""
"""
Write in epsilon and m values to test.
The parameters are explained here:

1. epsilon_list = This list should contain each epsilon value
for which you want to compute the minimum # of m rollouts you
need for epsilon convergence in cost.

2. num_simulations = We want a high probability bound, so each
value of m that we test should be run a number of times.

3. delta = This is the actual bound for the high probability.
So, we want our test to succeed proportion=delta times.

4. max_iter = max number of iterations for GD.

5. step_size = step size for GD.
"""
"""""""""""""""""""""""
"""""""""""""""""""""""

#lol i computed the biggest r value for epsilon = 0.001, this is below.
#it is SIGNIFICANTLY smaller than the r value of 0.005 i was using below.
r = 0.0007715
delta = 0.9
num_simulations = 1
epsilon_list = compute_epsilon_list(10**(-5), 10**(-12), 20)
epsilon_list = [0.001]
#m_test_list = compute_m_test_list(1, 1000, 20)
m_test_list = [200]
step_size = 0.11
max_iter = 1000000
print("Step Size: " + str(step_size))
print("Max Iterations: " + str(max_iter))

print(iterate_over_epsilon(epsilon_list, m_test_list))

def grid_search(start, end, difference):
	for step_size in np.arange(start, end, difference):
		print("Step Size: " + str(step_size))
		iterates = gradient_descent(K, step_size, 0.0001, max_iter, K_star, 100)
		plt.plot(range(0, len(iterates)), iterates)
		plt.show()

#grid_search(0.1, 0.12, 0.01)

#try different matrices, in particular huge/tiny values for Q and R
#and different A and B

#ALSO IMPORTANT: plot out the cost landscape for different values of
#A and B and Q and R, Ashwin will ask about what changing these does
#for us. also find the optimal K_star expression lol - he will ask about
#that as well. i think this will also help with my issue, currently the
#landscape is very steep so a step size small enough to overcome the initial
#fat gradient will overcome all future ones, hence no bouncing. but if we make
#the landscape shallower, then this is not true

#looks like 0.11 is as big as it gets for initial 0.9 (and initial 0.2). If I push the starting point further away
#then I hop across once but otherwise descent is smooth


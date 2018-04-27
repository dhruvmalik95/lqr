import numpy as np
import matplotlib.pyplot as plt
import math
import control
import scipy as sc

"""
This file is used to find dependency between the number of rollouts
required to get epsilon convergence in gradient.

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

K = 0.9

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
	numerator = (Q + K*R*K)*1 #instead of *1 it is *E(x^2)
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

"""""""""""""""""""""""
"""""""""""""""""""""""
"""Functions to find m dependence on epsilon"""
"""""""""""""""""""""""
"""""""""""""""""""""""

def iterate_over_epsilon(epsilon_list, m_test_list):
	"""Iterates over input list of epsilons and returns a list
	containing the minimum value of m required to get epsilon
	convergence in gradient for each epsilon."""
	true_gradient = compute_gradient(A, B, Q, R, K)
	minimum_m_list = []
	
	for epsilon in epsilon_list:
		print("current epsilon: " + str(epsilon))
		m = find_minimum_m(epsilon, m_test_list, true_gradient)
		minimum_m_list.append(m)

	return minimum_m_list

def find_minimum_m(epsilon, m_test_list, true_gradient):
	"""Given a fixed value of epsilon, this returns the minimum
	value of m required to get epsilon convergence in gradient with
	probability at least delta."""
	for m in m_test_list:
		print("current m: " + str(m))
		if hypothesis_test(K, epsilon, m, num_simulations, true_gradient) == True:
			return m
	print("m range too small")
	return 0

def hypothesis_test(K, epsilon, m, num_simulations, true_gradient):
	"""Given a fixed value of epsilon and fixed value of m to test,
	this function returns a boolean of whether this value of m ensures
	epsilon convergence, with proportion at least delta, out of all the
	num_simulations it runs."""
	num_correct = 0
	for _ in range(0, num_simulations):
		estimated_gradient = compute_estimated_gradient(A, B, Q, R, K, m)
		if abs(estimated_gradient - true_gradient) < epsilon:
			num_correct = num_correct + 1
	return (num_correct/num_simulations) > delta

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
"""
Write in epsilon and m values to test.
The parameters are explained here:

1. epsilon_list = This list should contain each epsilon value
for which you want to compute the minimum # of m rollouts you
need for epsilon convergence in gradient.

2. num_simulations = We want a high probability bound, so each
value of m that we test should be run a number of times.

3. delta = This is the actual bound for the high probability.
So, we want our test to succeed proportion=delta times.

"""
"""""""""""""""""""""""
"""""""""""""""""""""""

epsilon_list = compute_epsilon_list(0.1, 0.4, 20)
m_test_list = compute_m_test_list(1000, 10000, 20)

print("Epsilon List: " + str(epsilon_list))
print("M List To Test: " + str(m_test_list))

delta = 0.9
num_simulations = 100

#print(iterate_over_epsilon(epsilon_list, m_test_list))

"""""""""""""""""""""""
"""""""""""""""""""""""
"""Functions to compute dependence between epsilon and r."""
"""""""""""""""""""""""
"""""""""""""""""""""""

def check_r(r, epsilon, A, B, Q, R, K, true_gradient):
	estimate = (compute_actual_cost(A, B, Q, R, K + r) - compute_actual_cost(A, B, Q, R, K - r))/(2*r)
	return abs(estimate - true_gradient) < epsilon

def grid_search(start, diff, epsilon, A, B, Q, R, K):
	i = 0
	true_gradient = compute_gradient(A, B, Q, R, K)
	while i < 100000000:
		#print(i)
		current_r = start + i*diff
		if not check_r(current_r, epsilon, A, B, Q, R, K, true_gradient):
			print("success")
			return current_r - diff
		i = i + 1

def iterate_over_epsilon_2(epsilon_list, A, B, Q, R, K):
	r_list = []
	for epsilon in epsilon_list:
		print(epsilon)
		max_r = grid_search(start, diff, epsilon, A, B, Q, R, K)
		r_list.append(max_r)
	return r_list

start = 0.00001
diff = 0.000000001

print("Testing r vs. epsilon")
epsilon_list = compute_epsilon_list(0.00001, 0.1, 50)
epsilon_list = [0.001]
print("Epsilon List: " + str(epsilon_list))
print("Max r List: " + str(iterate_over_epsilon_2(epsilon_list, A, B, Q, R, 4.0)))

def shit(A, B, Q, R):
	a = A*B*R
	b = B*B*Q + R - A*A*R
	c = -1*B*Q*A
	return np.roots([a,b,c])

#print(shit(A, B, Q, R))



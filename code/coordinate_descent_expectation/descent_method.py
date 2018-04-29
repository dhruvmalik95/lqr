import numpy as np
import matplotlib.pyplot as plt
import math
import control
import scipy as sc

def compute_optimal_controller(A, B, Q, R):
	return control.dare(A, B, Q, R)[2]

def is_pos_def(x):
	return np.all(np.linalg.eigvals(x) > 0)

def is_stable(A, B, K):
	M = A - B.dot(K)
	eigenvalues = np.linalg.eigvals(M)
	return np.all(np.absolute(eigenvalues) < 1)

def compute_new_state(A, B, K, current_state):
	current_control = K.dot(current_state)
	return A.dot(current_state) - B.dot(current_control)

def compute_current_cost(Q, R, K, current_state):
	# note we are using Q_d
	current_control = K.dot(current_state)
	return current_state.dot(np.multiply(Q_d, current_state)) + current_control.dot(np.multiply(R_d, current_control))

def compute_controller_cost(A, B, Q, R, K, epsilon, max_iter, initial_state):
	current_state = initial_state
	i = 0
	old_total = -10
	total = 0
	
	while i < max_iter and (total - old_total) > epsilon:
		#print(i)
		current_cost = compute_current_cost(Q, R, K, current_state)
		old_total = total
		total = total + current_cost
		current_state = compute_new_state(A, B, K, current_state)
		i = i + 1
	return total

def compute_controller_gaussian_cost(A, B, Q, R, K, epsilon, max_iter, d):
	total = 0
	for state in state_list:
		total = total + compute_controller_cost(A, B, Q, R, K, epsilon, max_iter, np.array(state))
	return total/d

def compute_singular_values(M):
	return np.linalg.svd(M)[1]

def compute_operator_norm(M):
	return max(compute_singular_values(M))

def compute_min_singular_value(M):
	return min(compute_singular_values(M))

def distance_frobenius(x, y):
	return abs(np.linalg.norm(x, ord='fro') - np.linalg.norm(y, ord='fro'))

def converged(true, estimate, epsilon):
	#true_cost = compute_controller_gaussian_cost(A, B, Q, R, true, cost_tol, cost_max_iter, d)
	estimate_cost = compute_controller_gaussian_cost(A, B, Q, R, estimate, cost_tol, cost_max_iter, d)
	return abs(K_star_cost - estimate_cost)/K_star_cost < epsilon

def sample_uniform_perturbation(d, r):
	uniform_matrix = np.random.normal(size=(d,d))
	return r*uniform_matrix*(1/np.linalg.norm(uniform_matrix, ord='fro'))

def estimate_gradient(d, r, m, K):
	# create function input parameter for tolerance
	tolerance = 0.0001
	
	right = compute_controller_gaussian_cost(A, B, Q, R, K, cost_tol, cost_max_iter, d)
	total = np.zeros(shape=(d,d))
	for _ in range(1, m+1):
		#print(_)
		U = sample_uniform_perturbation(d, 1)
		left = compute_controller_gaussian_cost(A, B, Q, R, K + r*U, cost_tol, cost_max_iter, d)
		#right = compute_controller_gaussian_cost(A, B, Q, R, K, cost_tol, cost_max_iter, d)
		K_hat = ((left - right)/r)*U
		total = total + K_hat
	return (1/m)*total

def sample_row_column(d):
	row = np.random.randint(low=0, high=d)
	col = np.random.randint(low=0, high=d)
	return row, col

def estimate_partial_derivative(d, r, m, K):
	row, col = sample_row_column(d)
	perturbation_left = np.zeros(shape=(d,d))
	perturbation_left[row][col] = r
	perturbation_right = np.zeros(shape=(d,d))
	perturbation_right[row][col] = -1*r

	K_left = K + perturbation_left
	K_left_cost = compute_controller_gaussian_cost(A, B, Q, R, K_left, cost_tol, cost_max_iter, d)
	K_right = K + perturbation_right
	K_right_cost = compute_controller_gaussian_cost(A, B, Q, R, K_right, cost_tol, cost_max_iter, d)

	derivative_value = (K_left_cost - K_right_cost)/(2*r)
	derivative = np.zeros(shape=(d,d))
	derivative[row][col] = derivative_value

	return derivative

def gradient_step(current_K, step_size, gradient):
	new_K = current_K - step_size*gradient
	return new_K

def gradient_descent(d, r, m, epsilon, step_size, initial_K, optimal_K):
	current_K = initial_K
	N = 0
	stop = 100000
	while N < stop and not converged(K_star, current_K, epsilon):
		#print("Current K: " + str(current_K))
		print("Current K Cost: " + str(compute_controller_gaussian_cost(A, B, Q, R, current_K, cost_tol, cost_max_iter, d)))
		print(N)
		print("K_star Cost: " + str(K_star_cost))
		gradient_estimate = estimate_partial_derivative(d, r, m, current_K)
		#print("Gradient Estimate: " + str(gradient_estimate))
		#print("Norm Of Gradient Estimate: " + str(np.linalg.norm(gradient_estimate, ord='fro')))
		current_K = gradient_step(current_K, step_size/(500000), gradient_estimate)
		if is_stable(A, B, current_K) == False:
			print("Stopped Early - Unstable: " + str(N))
			N = stop + 1
			return "fuck unstable"
		N = N + 1
		#print("Current K: " + str(current_K))
	return N

def generate_initial_states(d):
	state_list = []
	for i in range(0, d):
		zero_list = np.zeros(d)
		zero_list[i] = 1
		state_list.append(zero_list)
	return state_list

cost_tol = 0.0000001
cost_max_iter = 100000

d = 2
A = 0.05*np.array([[-10, 0], [7, 9]])
B = 0.05*np.array([[-8, 2], [3, 1]])
Q = 0.05*np.array([[58, 2], [2, 58]])
R = 0.05*np.array([[9, 5], [5, 9]])
Q_d = np.diag(Q)
R_d = np.diag(R)
print(Q_d)
state_list = generate_initial_states(d)


# """
# TEST MATRIX
# """
# print("d: " + str(d))
# print("A: " + str(A))
# print("B: " + str(B))
# print("Q: " + str(Q))
# print("R: " + str(R))
# print("Initial State List: " + str(state_list))
# state_list = generate_initial_states(d)

# r = 1e-12
# """Manipulate this"""
# #m = 10*(d)
# m = 0 #doesn't matter here
# epsilon = 0.1
# step_size = 0.5

# print("Q is PSD: " + str(is_pos_def(Q)))
# print("R is PSD: " + str(is_pos_def(R)))

# K_star = np.array(compute_optimal_controller(A, B, Q, R))
# print("K_star: " + str(K_star))
# K_star_cost = compute_controller_gaussian_cost(A, B, Q, R, K_star, cost_tol, cost_max_iter, d)
# print("K_star Cost: " + str(K_star_cost))

# initial_K = np.array([[1.6, 1.138],[1.5, 5.3]])
# print("Initial K: " + str(initial_K))
# print("Initial K Is Stable: " + str(is_stable(A, B, initial_K)))
# initial_K_cost = compute_controller_gaussian_cost(A, B, Q, R, initial_K, cost_tol, cost_max_iter, d)
# print("Initial K Cost: " + str(initial_K_cost))

# gradient_descent(d, r, m, epsilon, step_size, initial_K, K_star)

""""""""""""""""""""""""""

def generate_transitions(d):
	"""Returns random A and B sampled from uniform distribution
	over [-10,10]"""
	return np.random.randint(low=-10, high=10, size=(d, d)), np.random.randint(low=-10, high=10, size=(d, d))

def generate_rewards(d):
	"""Returns random diagonal Q and R"""
	v = np.random.randint(low=1, high=10, size=d)
	Q = np.diag(v)
	
	w = np.random.randint(low=1, high=10, size=d)
	R = np.diag(w)
	return Q, R, v, w

def generate_initial_K(d, A, B, K_star):
	perturbation = 0.75*(np.random.random(size=(d,d)) - 0.5)/(10/6)
	K = np.copy(K_star) + perturbation
	"""Introduce Operator Norm Condition <0.5 here - maybe divide by d to help generation??"""
	while not is_stable(A, B, K):
		perturbation = 0.75*(np.random.random(size=(d,d)) - 0.5)/(10/0.21)
		K = np.copy(K_star) + perturbation
	return K

def compute_d_test_list(lower, upper, density):
	difference = (np.log(upper) - np.log(lower)) / density
	print(difference)
	l = []
	for i in np.arange(np.log(lower), np.log(upper), difference):
		l.append(int(round(np.exp(i))))
	return l

#d_test_list = compute_d_test_list(2, 100, 10)
d_test_list = range(64,65)

epsilon_list = []
epsilon_worst_list = []

d_number = 1
initialize_number = 1
"""
for d in d_test_list:
	epsilon_avg = 0
	e_worst_list = []
	for __ in range(0, d_number):
		print("Iteration: " + str(__))
		A, B = generate_transitions(d)
		A = 0.3*A #maybe divide these numbers by log(d)?
		B = 0.3*B
		Q, R, Q_d, R_d = generate_rewards(d)
		Q = 1*Q
		R = 1*R
		state_list = generate_initial_states(d)
		cost_tol = 0.0001
		cost_max_iter = 100000

		r = 1e-12
		"""  """
		m = 0 #doesn't matter here
		epsilon = 0.5
		step_size = 1.0 # remember im dividing this by initial cost, and it may be too big even for perfect gradients

		print("d: " + str(d))
		print("A: " + str(A))
		print("B: " + str(B))
		print("Q: " + str(Q))
		print("R: " + str(R))
		print("Initial State List: " + str(state_list))

		print("Q is PSD: " + str(is_pos_def(Q)))
		print("R is PSD: " + str(is_pos_def(R)))

		K_star = np.array(compute_optimal_controller(A, B, Q, R))
		print("K_star: " + str(K_star))
		K_star_cost = compute_controller_gaussian_cost(A, B, Q, R, K_star, cost_tol, cost_max_iter, d)
		print("K_star Cost: " + str(K_star_cost))
		
		epsilon_avg_initialize = 0
		for _ in range(0, initialize_number):
			initial_K = generate_initial_K(d, A, B, K_star)
			# initial_K = np.array([[1.6, 1.138],[1.5, 5.3]])


			print("Initial K: " + str(initial_K))
			print("Initial K Is Stable: " + str(is_stable(A, B, initial_K)))
			initial_K_cost = compute_controller_gaussian_cost(A, B, Q, R, initial_K, cost_tol, cost_max_iter, d)
			print("Initial K Cost: " + str(initial_K_cost))
			# print(find_direction(3, 2, 0.001, initial_K))
			# print(descend_hill(3, 2, 0.001, initial_K))
			
			#step_size = epsilon/(d*d)

			estimate = gradient_descent(d, r, m, epsilon, step_size, initial_K, K_star)
			epsilon_avg_initialize = estimate
		e_worst_list.append(estimate)
		epsilon_avg_initialize = epsilon_avg_initialize/initialize_number
		epsilon_avg = epsilon_avg + epsilon_avg_initialize
	
	epsilon_avg = epsilon_avg/d_number
	epsilon_list.append(epsilon_avg)
	epsilon_worst_list.append(min(e_worst_list))
	print("____________________________")


print("Epsilon List: " + str(epsilon_list))
print("Epsilon Worst List: " + str(epsilon_worst_list))
plt.plot(np.log(d_test_list), np.log(epsilon_list/step_size))
plt.show()
"""
x = np.log([2,4,8,16,32,64])
y = np.log([500,1000,5000,15000,35000,300000])
print(sc.stats.linregress(x, y))
plt.scatter(x, y)
plt.xlabel('log(dimension)')
plt.ylabel('log(# of iterations)')
plt.show()
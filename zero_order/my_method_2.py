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
	current_control = K.dot(current_state)
	return current_state.dot(Q_d.dot(current_state)) + current_control.dot(R_d.dot(current_control))

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
	if not is_stable(A, B, K):
		return np.inf
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

def generate_initial_states(d):
	state_list = []
	for i in range(0, d):
		zero_list = np.zeros(d)
		zero_list[i] = 1
		state_list.append(zero_list)
	return state_list

def get_coordinates(index, d):
	row = index // d
	column = index % d
	return row, column

def minimize_coordinate(index, d, K, epsilon):
	row, column = get_coordinates(index, d)
	if find_direction(index, d, step_size, K) > 0:
		x1 = K[row][column]
		x2 = x1 + 20
		return bisection_helper(row, column, K, x1, x2, epsilon)
	else:
		x1 = K[row][column]
		x2 = x1 - 20
		return bisection_helper(row, column, K, x1, x2, epsilon)

def bisection_helper(row, column, K, a, b, epsilon):
	if abs(a - b) < epsilon:
		Ka = np.copy(K)
		Kb = np.copy(K)
		Ka[row][column] = a
		Kb[row][column] = b
		if compute_controller_gaussian_cost(A, B, Q, R, Ka, cost_tol, cost_max_iter, d) < compute_controller_gaussian_cost(A, B, Q, R, Kb, cost_tol, cost_max_iter, d):
			return Ka
		else:
			return Kb

	x_1 = a
	x_2 = b
	x3 = (x1 + x2)/2
	x4 = (x1 + x3)/2
	x5 = (x2 + x3)/2

	K1 = np.copy(K)
	K1[row][column] = x1
	K2 = np.copy(K)
	K2[row][column] = x2
	K3 = np.copy(K)
	K3[row][column] = x3
	K4 = np.copy(K)
	K4[row][column] = x4
	K5 = np.copy(K)
	K5[row][column] = x5

	if compute_controller_gaussian_cost(A, B, Q, R, K1, cost_tol, cost_max_iter, d) < compute_controller_gaussian_cost(A, B, Q, R, K4, cost_tol, cost_max_iter, d):
		a = x1
		b = x4
	else:
		if compute_controller_gaussian_cost(A, B, Q, R, K4, cost_tol, cost_max_iter, d) < compute_controller_gaussian_cost(A, B, Q, R, K3, cost_tol, cost_max_iter, d):
			a = x1
			b = x3
		else:
			if compute_controller_gaussian_cost(A, B, Q, R, K3, cost_tol, cost_max_iter, d) < compute_controller_gaussian_cost(A, B, Q, R, K5, cost_tol, cost_max_iter, d):
				a = x1
				b = x5
			else:
				a = x4
				b = x2

	return bisection_helper(row, column, K, a, b)

def take_step(step, row, column, K):
	#print(K)
	current_entry = K[row][column]
	#print(current_entry)
	K[row][column] = current_entry + step
	return K

def find_direction(index, d, step_size, K):
	row, column = get_coordinates(index, d)
	# print("r:"+str(row))
	# print("c:"+str(column))
	left_K = np.copy(K)
	right_K = np.copy(K)
	
	right_K = take_step(step_size, row, column, right_K)
	left_K = take_step(-1*step_size, row, column, left_K)
	right_K_cost = compute_controller_gaussian_cost(A, B, Q, R, right_K, cost_tol, cost_max_iter, d)
	left_K_cost = compute_controller_gaussian_cost(A, B, Q, R, left_K, cost_tol, cost_max_iter, d)
	
	# print(left_K)
	# print(right_K)
	# print(left_K_cost)
	# print(right_K_cost)
	if right_K_cost < left_K_cost:
		return step_size
	else:
		return -1*step_size

def solve(d, step_size, K_star, K, epsilon):
	count = 0
	#while not converged(K_star, K, epsilon):
	old_cost = -10
	stationary = False
	while not stationary: #not necessarily close to optimum, but it has reached stationary
		current_cost = compute_controller_gaussian_cost(A, B, Q, R, K, cost_tol, cost_max_iter, d)
		for i in range(0, d*d):
			#print(i)
			row, column = get_coordinates(i, d)
			K = minimize_coordinate(i, d, K, epsilon)
		count = count + 1
		if current_cost - old_cost == 0:
			stationary = True
		if current_cost > old_cost and old_cost != -10:
			#this happened, especially with d=10, check this
			print("NOT MONOTONIC")
			return K
		# print(K)
		# print("Estimated Optimum Cost: " + str(current_cost))
		# print(count)

		old_cost = current_cost
	
	#print("Estimated Optimum K: " + str(K))
	return K

# A = np.array([[-10, 0], [7, 9]])
# B = np.array([[-8, 2], [3, 1]])
# Q = np.array([[58, 2], [2, 58]])
# R = np.array([[9, 5], [5, 9]])

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
	perturbation = 0.75*(np.random.random(size=(d,d)) - 0.5)/(10)
	K = np.copy(K_star) + perturbation
	while not is_stable(A, B, K):
		perturbation = 0.75*(np.random.random(size=(d,d)) - 0.5)/(10)
		K = np.copy(K_star) + perturbation
	return K

def compute_d_test_list(lower, upper, density):
	difference = (np.log(upper) - np.log(lower)) / density
	print(difference)
	l = []
	for i in np.arange(np.log(lower), np.log(upper), difference):
		l.append(int(round(np.exp(i))))
	return l

d_test_list = compute_d_test_list(2, 100, 10)
print("D Test List: " + str(d_test_list))

#d_test_list = [136]

epsilon_list = []
epsilon_worst_list = []

d_number = 25
initialize_number = 1

"""Make Q and R diagonal matrices - perhaps write my own method for this
in case numpy is not optimized for diagonal multiplication - np.diag?

this is leads to d^4 (it doesnt really cuz of transitions, tell ashwin this),
so we can simulate d=2^7, so 7 (or more) points on the x axis
run each one ~10 times

set the A-BK to be less than 0.5

metric should be RELATIVE error in COST"""

# for d in d_test_list:
# 	epsilon_avg = 0
# 	e_worst_list = []
# 	for __ in range(0, d_number):
# 		print("Iteration: " + str(__))
# 		A, B = generate_transitions(d)
# 		A = 0.01*A
# 		B = 0.01*B
# 		Q, R, Q_d, R_d = generate_rewards(d)
# 		Q = 0.1*Q
# 		R = 0.1*R
# 		state_list = generate_initial_states(d)
# 		cost_tol = 0.00000001
# 		cost_max_iter = 100000

# 		print("d: " + str(d))
# 		print("A: " + str(A))
# 		print("B: " + str(B))
# 		print("Q: " + str(Q))
# 		print("R: " + str(R))
# 		print("Initial State List: " + str(state_list))

# 		print("Q is PSD: " + str(is_pos_def(Q)))
# 		print("R is PSD: " + str(is_pos_def(R)))

# 		K_star = np.array(compute_optimal_controller(A, B, Q, R))
# 		print("K_star: " + str(K_star))
# 		print("K_star Cost: " + str(compute_controller_gaussian_cost(A, B, Q, R, K_star, cost_tol, cost_max_iter, d)))

# 		epsilon_avg_initialize = 0
# 		for _ in range(0, initialize_number):
# 			initial_K = generate_initial_K(d, A, B, K_star)
# 			# initial_K = np.array([[1.6, 1.138],[1.5, 5.3]])


# 			print("Initial K: " + str(initial_K))
# 			print("Initial K Is Stable: " + str(is_stable(A, B, initial_K)))
#			initial_K_cost = compute_controller_gaussian_cost(A, B, Q, R, initial_K, cost_tol, cost_max_iter, d)
# 			# print("Initial K Cost: " + str(initial_K_cost))
# 			# print(find_direction(3, 2, 0.001, initial_K))
# 			# print(descend_hill(3, 2, 0.001, initial_K))
			
# 			#step_size = epsilon/(d*d)
# 			step_size = 0.005
# 			epsilon = 10000 #this value doesnt matter since we are trying to get epislon

# 			estimate = solve(d, step_size, K_star, initial_K, epsilon)
# 			epsilon_avg_initialize = epsilon_avg_initialize + distance_frobenius(K_star, estimate)
# 		e_worst_list.append(distance_frobenius(K_star, estimate))
# 		epsilon_avg_initialize = epsilon_avg_initialize/initialize_number
# 		epsilon_avg = epsilon_avg + epsilon_avg_initialize
	
# 	epsilon_avg = epsilon_avg/d_number
# 	epsilon_list.append(epsilon_avg)
# 	epsilon_worst_list.append(min(e_worst_list))
# 	#print("Count: " + str(solve(d, step_size, K_star, initial_K, epsilon)))
# 	#[0.010284003748980778, 0.012191519761673408, 0.0083251628412912646, 0.0080304367926726551, 0.0059513361458982238, 0.0056990577208198452, 0.0058746699764256676, 0.0040847669712047365, 0.004455948266739629]
# 	print("____________________________")

step_size = 0.005

print("Epsilon List: " + str(epsilon_list))
print("Epsilon Worst List: " + str(epsilon_worst_list))
plt.plot(np.log(d_test_list), np.log(epsilon_list/step_size))
plt.show()

"""Pathological Example 1"""
# cost_tol = 0.000001
# cost_max_iter = 100000

# epsilon = 0.00005
# step_size = epsilon/(2*2)

# A = np.array([[-7,  -4], [-10, -10]])
# B = np.array([[4, -8], [4, -2]])
# Q = np.array([[433, 57], [57, 145]])
# R = np.array([[377, 82], [82, 100]])

# K_star = np.array([[-2.38668915, -2.58301383], [-0.20677211, -0.66335369]])
# initial_K = np.array([[-2.34020455, -2.52811338], [-0.19770009, -0.59259569]])
# print("Q is PSD: " + str(is_pos_def(Q)))
# print("R is PSD: " + str(is_pos_def(R)))
# print("Initial K Is Stable: " + str(is_stable(A, B, initial_K)))
# solve(2, step_size, K_star, initial_K, epsilon)
# """
# the step size is epsilon/(2*2), since we have 2*2 entries
# with epislon=0.05, we converge to [[-2.39020455 -2.51561338]
#  [-0.21020009 -0.60509569]]
# with epislon=0.005, we converge to [[-2.37395455 -2.51311338]
#  [-0.19645009 -0.60634569]]
# with epsilon=0.0005, we converge to [[-2.37270455 -2.57186338]
#  [-0.19545009 -0.65434569]]
# with epislon=0.00005, we converge to [[-2.38561705 -2.58245088]
#  [-0.20590009 -0.66289569]]
# with epsilon=0.000005, we converge to [[-2.38658205 -2.58296088]
#  [-0.20668509 -0.66331069]]
# with epsilon=0.0000005, we converge to [[-2.38667868 -2.58300876]
#  [-0.20676359 -0.66334957]]

# # none of these are epsilon convergence in Frobenius norm, but are all
# # closer and closer to K_star

# # here's the other important thing to note: i plotted the marginal cost
# # by manipulating each of the 4 indices, and every single plot turned out
# # to look completely convex.
# # """

"""Pathological Example 2"""
# cost_tol = 0.000001
# cost_max_iter = 100000

# epsilon = 0.001
# step_size = epsilon/(2*2)

# A = np.array([[8, 8], [2, 8]])
# B = np.array([[5, -8], [4, -6]])
# Q = np.array([[64, 8], [8, 26]])
# R = np.array([[40, -34], [-34, 29]])

# K_star = np.array([[-10.37665506, 4.1193407], [-7.65848126, 1.69405854]])
# initial_K = np.array([[-10.33011151, 4.1123955], [-7.66463932, 1.74909735]])
# print("Q is PSD: " + str(is_pos_def(Q)))
# print("R is PSD: " + str(is_pos_def(R)))
# print("Initial K Is Stable: " + str(is_stable(A, B, initial_K)))
# solve(2, step_size, K_star, initial_K, epsilon)
# """
# the step size is epsilon/(2*2), since we have 2*2 entries
# with epislon=0.005, we converge to [[-10.32136151   4.1648955 ]
#  [ -7.62588932   1.72159735]]
# with epislon=0.0005, we converge to [[-10.35998651   4.1667705 ]
#  [ -7.64851432   1.72234735]], which actually gives epsilon convergence in Froebenius norm
# """

def plot_landscape(d):
	current_K = initial_K
	l = []
	while is_stable(A, B, current_K):
		l.append(compute_controller_gaussian_cost(A, B, Q, R, current_K, cost_tol, cost_max_iter, d))
		#print("hi")
		current_K = take_step(step_size, 1, 1, np.copy(current_K))
		#print(max(np.linalg.eigvals(A-B.dot(current_K))))
	plt.plot(range(0, len(l)), np.log(l))
	plt.show()

#plot_landscape(2)


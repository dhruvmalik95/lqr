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
	return current_state.dot(Q.dot(current_state)) + current_control.dot(R.dot(current_control))

def compute_controller_cost(A, B, Q, R, K, epsilon, max_iter, initial_state):
	current_state = initial_state
	i = 0
	old_total = -10
	total = 0
	
	while i <= max_iter and (total - old_total) > epsilon:
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

def compute_state_correlation_matrix(A, B, Q, R, K, epsilon, max_iter, d):
	t = np.zeros(shape=(d,d))
	for state in state_list[:]:
		m = np.zeros(shape=(d,d)) + np.outer(state, state)
		state = compute_new_state(A, B, K, state)
		for i in range(0, max_iter):
			m = m + np.outer(state, state)
			state = compute_new_state(A, B, K, state)
		t = t + m
	return t/d

def compute_singular_values(M):
	return np.linalg.svd(M)[1]

def compute_operator_norm(M):
	return max(compute_singular_values(M))

def compute_min_singular_value(M):
	return min(compute_singular_values(M))

def distance_frobenius(x, y):
	return abs(np.linalg.norm(x, ord='fro') - np.linalg.norm(y, ord='fro'))

def converged(true, estimate, epsilon):
	return distance_frobenius(true, estimate) < epsilon


d = 3

A = np.array([ [1,0,0], [0,1,0], [0,0,1] ])
B = np.array([ [1,0,0], [0,1,0], [0,0,1] ])
Q = np.array([ [1,0,0], [0,1,0], [0,0,1] ])
R = np.array([ [1,0,0], [0,1,0], [0,0,1] ])

K1 = np.array([ [1,0,-10], [-1,1,0], [0,0,1] ])
K2 = np.array([ [1,-10,0], [0,1,7], [-1,0,1] ])
K_star = np.array(compute_optimal_controller(A, B, Q, R))

state_list = [ [1,0,0], [0,1,0], [0,0,1] ]

print(A)
print(B)
print(Q)
print(R)
print("_____")
print(K1)
print(K2)
print("_____")

s = compute_state_correlation_matrix(A, B, Q, R, K_star, 0.00001, 15, d)
K_perturbed = K_star + np.random.uniform(low=-0.1, high=0.1, size=(d,d))
s_p = compute_state_correlation_matrix(A, B, Q, R, K_perturbed, 0.00001, 15, d)

print("_____")
big = True
while big:
	vec = np.random.normal(size=d)
	vec = vec/np.linalg.norm(vec)
	if vec.dot(s.dot(vec)) > vec.dot(s_p.dot(vec)):
		big = False
		print(s)
		print(s_p)
		print(vec)
		print(np.linalg.norm(vec))
		print(vec.dot(s.dot(vec)))
		print(vec.dot(s_p.dot(vec)))
		print("done")

#Kakade:
# d = 3

# A = np.array([ [1,0,0], [0,1,0], [0,0,1] ])
# B = np.array([ [1,0,0], [0,1,0], [0,0,1] ])
# Q = np.array([ [1,0,0], [0,1,0], [0,0,1] ])
# R = np.array([ [1,0,0], [0,1,0], [0,0,1] ])

# K1 = np.array([ [1,0,-10], [-1,1,0], [0,0,1] ])
# K2 = np.array([ [1,-10,0], [0,1,0], [-1,0,1] ])

# state_list = [ [1,0,0], [0,1,0], [0,0,1] ]

# print(A)
# print(B)
# print(Q)
# print(R)
# print("_____")
# print(K1)
# print(K2)
# print("_____")
# print(np.linalg.eigvals(A - B.dot(K1)))
# print(np.linalg.eigvals(A - B.dot(K2)))
# avg = (K1+K2)/2
# print(np.linalg.eigvals(A - B.dot(avg)))

# print(compute_controller_gaussian_cost(A, B, Q, R, K1, 0.000001, 3, d))
# print(compute_controller_gaussian_cost(A, B, Q, R, K2, 0.000001, 3, d))
# print(compute_controller_gaussian_cost(A, B, Q, R, avg, 0.000001, 3, d))



import numpy as np
import matplotlib.pyplot as plt

n=20

c = np.ones((n,1))
x = np.zeros((n,1))

lr = 1

t_k = 0
eps = 1e-5

def calculate_hessian(x):
    gradient = 2*x/(1 - x**2)
    hessian = np.diag(np.ravel(1/(1+x)**2 + 1/(1-x)**2))
    return gradient, hessian

def calculate_dual_norm(g, H):
    return np.sum(np.sqrt(g.T @ np.linalg.inv(H) @ g))

gradient, hessian = calculate_hessian(x)
beta = 2

v = 2*n

f_vals = []
while t_k < (v + beta*(beta+np.sqrt(v))/(1-beta))/eps:
    gradient, hessian = calculate_hessian(x)
    t_k = t_k + lr/calculate_dual_norm(c, hessian)
    lambda_k = calculate_dual_norm(t_k*c + gradient, hessian)
    xi_k = lambda_k**2/(1+lambda_k)
    x = x - 1/(1+xi_k)*np.linalg.inv(hessian)@(t_k*c + gradient)
    f_vals.append(np.ravel(c.T@x)+n)

print("Optimal:", x)

plt.loglog(np.arange(len(f_vals)), f_vals)
plt.title("CP Log-Barrier lr=%s, eps=%s" % (lr, eps))
plt.xlabel("Iterations")
plt.ylabel("Suboptimality Gap")
plt.show()
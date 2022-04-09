import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Initialize data.
d = 10

kmax = 5
MAX_ITERS = 170
global_opt = -0.841408

A = np.zeros((kmax, d, d))
b = np.zeros((kmax, d))
l = 0.5

for k in range(kmax):
    for i in range(d):
        running_sum = 0
        for j in range(d):
            if i == j:
                continue
            elif i < j:
                A[k,i,j] = np.exp((1+i)/(1+j))*np.cos((1+i)*(1+j))*np.sin(k+1)
                running_sum += np.abs(A[k,i,j])
            else:
                A[k,i,j] = np.exp((1+j)/(1+i))*np.cos((1+i)*(1+j))*np.sin(k+1)
                running_sum += np.abs(A[k,i,j])
        A[k,i,i] = (i+1)/10*np.abs(np.sin(k+1)) + running_sum

for k in range(kmax):
    for i in range(d):
        b[k, i] = np.exp((1+i)/(1+k))*np.sin((1+i)*(1+k))


x = np.ones((d,1))


x_list = []
f_val_list = []
gradient_list = []
running_opt = np.inf
opts_list = np.zeros(MAX_ITERS)

for i in range(MAX_ITERS):
    if i % 2 == 0:
        print(i)
    x_list.append(x)
    f_vals = np.squeeze(x.T @ A @ x) - np.squeeze(b@x)
    f_loc = np.argmax(f_vals)
    f_val = f_vals[f_loc]
    running_opt = min(f_val, running_opt)
    opts_list[i] = running_opt

    f_val_list.append(f_val)
    gradient_list.append(2*A[f_loc]@x - b[f_loc][:, None])
    X = cp.Variable((1,d))
    Z = cp.Variable()

    f_i = [Z >= f_val_list[i] + (X-x_list[i].T)@gradient_list[i] for i in range(len(x_list))]
    
    prob = cp.Problem(cp.Minimize(Z), f_i + [cp.norm1(X) <= d])
    prob.solve()

    level = (1-l)*prob.value + l*running_opt
    prob = cp.Problem(cp.Minimize(cp.norm1(X - x)), f_i + [Z <= level, cp.norm1(X) <= d])
    prob.solve()
    x = X.value.T

plt.loglog(np.arange(MAX_ITERS), opts_list - -0.841408)
plt.title("MAXQUAD Level Method lambda = %s" % l)
plt.xlabel("Iterations")
plt.ylabel("Suboptimality Gap")
plt.show()
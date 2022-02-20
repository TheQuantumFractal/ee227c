import numpy as np
import matplotlib.pyplot as plt

# Initialize data.
d = 10

kmax = 5

A = np.zeros((kmax, d, d))
b = np.zeros((kmax, d))

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
f_vals = np.squeeze(x.T @ A @ x) - np.squeeze(b@x)
f_loc = np.argmax(f_vals)
f_val = f_vals[f_loc]
running_opt = f_val
print("f(x1) = %s" % f_val)

# Regular step size
MAX_ITERS = 500000
C = 5e-2
opt = f_val

lr = C / np.sqrt(MAX_ITERS)

for i in range(MAX_ITERS):
    f_vals = np.squeeze(x.T @ A @ x) - np.squeeze(b@x)
    f_loc = np.argmax(f_vals)
    f_val = f_vals[f_loc]

    gradient = 2*A[f_loc]@x - b[f_loc][:, None]
    x -= lr*gradient/np.linalg.norm(gradient)
    opt = min(f_val, opt)

print("Global opt: %s" % opt)

MAX_ITERS = 100000
C = 5e-2

lr = C / np.sqrt(MAX_ITERS)

x = np.ones((d,1))

opts_list = np.zeros(MAX_ITERS)

for i in range(MAX_ITERS):
    f_vals = np.squeeze(x.T @ A @ x) - np.squeeze(b@x)
    f_loc = np.argmax(f_vals)
    f_val = f_vals[f_loc]

    subgradient = 2*A[f_loc]@x - b[f_loc][:, None]
    x -= lr*subgradient/np.linalg.norm(subgradient)
    running_opt = min(f_val, running_opt)
    opts_list[i] = running_opt

# Plots

plt.loglog(np.arange(MAX_ITERS), opts_list - opt)
plt.title("MAXQUAD Subgradient Method C = 5e-2")
plt.xlabel("Iterations")
plt.ylabel("Suboptimality Gap")
plt.show()

print("C=5e-2 opt: %s" % running_opt)

# Polyak Step Size
MAX_ITERS = 100000

x = np.ones((d,1))
f_vals = np.squeeze(x.T @ A @ x) - np.squeeze(b@x)
f_loc = np.argmax(f_vals)
f_val = f_vals[f_loc]
running_opt = f_val

opts_list = np.zeros(MAX_ITERS)

for i in range(MAX_ITERS):
    f_vals = np.squeeze(x.T @ A @ x) - np.squeeze(b@x)
    f_loc = np.argmax(f_vals)
    f_val = f_vals[f_loc]

    subgradient = 2*A[f_loc]@x - b[f_loc][:, None]

    suboptimality_gap = f_val - opt
    lr = suboptimality_gap/np.linalg.norm(subgradient)


    x -= lr*subgradient/np.linalg.norm(subgradient)
    running_opt = min(f_val, running_opt)
    opts_list[i] = running_opt

plt.loglog(np.arange(MAX_ITERS), opts_list - opt)
plt.title("MAXQUAD Subgradient Method Polyak Step Size")
plt.xlabel("Iterations")
plt.ylabel("Suboptimality Gap")
plt.show()
print("Polyak opt: %s" % running_opt)
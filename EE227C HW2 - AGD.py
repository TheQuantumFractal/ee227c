import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
A = np.random.normal(size=(20,20))

x = np.ones((20,1))

MAX_ITERS = 2000
lr = 5e-3
best_list = []
best = np.inf

for i in range(MAX_ITERS):
    best = min(best, np.linalg.norm(A@x))
    best_list.append(best)
    gradient = 2*(A.T @ A)@x
    x -= gradient*lr
    # print(x)
best = min(best, np.linalg.norm(A@x))
best_list.append(best)
plt.plot(np.arange(MAX_ITERS+1), best_list, label="Gradient Descent lr=%s" % lr)
# plt.title("Gradient Descent lr=%s" % lr)
plt.xlabel("Iterations")
plt.ylabel("Value of Best Iterate")
# plt.show()

best_list2 = []
best = np.inf
x = np.ones((20,1))
z = x
lambda_t = 1
gamma_t = 0
for i in range(MAX_ITERS):
    best = min(best, np.linalg.norm(A@x))
    best_list2.append(best)
    y_t = (1-gamma_t)*x + gamma_t*z
    gradient = lr*2*(A.T @ A)@y_t
    # print(y_t)
    z -= gamma_t/lambda_t*gradient
    x = y_t - gradient
    if i > 3:
        gamma_t = 2/i
        lambda_t = 6/i/(i+1)
    # print(x)
best = min(best, np.linalg.norm(A@x))
best_list2.append(best)
plt.loglog(np.arange(MAX_ITERS+1), best_list2, label="AGD lr=%s" % lr)
# plt.title("Gradient Descent lr=%s" % lr)
plt.xlabel("Iterations")
plt.ylabel("Value of Best Iterate")
leg = plt.legend(loc='upper center')
plt.show()

A = np.zeros((20,20))

x = np.ones((20,1))
for i in range(A.shape[0]):
    A[i][i] = 2
    if i > 0:
        A[i][i-1] = -1
    if i < A.shape[0] - 1:
        A[i][i+1] = -1

MAX_ITERS = 2000
lr = 5e-3
best_list = []
best = np.inf

for i in range(MAX_ITERS):
    best = min(best, np.linalg.norm(A@x))
    best_list.append(best)
    gradient = 2*(A.T @ A)@x
    x -= gradient*lr
    # print(x)
best = min(best, np.linalg.norm(A@x))
best_list.append(best)
plt.plot(np.arange(MAX_ITERS+1), best_list, label="Gradient Descent lr=%s" % lr)
# plt.title("Gradient Descent lr=%s" % lr)
plt.xlabel("Iterations")
plt.ylabel("Value of Best Iterate")
# plt.show()

best_list2 = []
best = np.inf
x = np.ones((20,1))
z = x
lambda_t = 1
gamma_t = 0
for i in range(MAX_ITERS):
    best = min(best, np.linalg.norm(A@x))
    best_list2.append(best)
    y_t = (1-gamma_t)*x + gamma_t*z
    gradient = lr*2*(A.T @ A)@y_t
    # print(y_t)
    z -= gamma_t/lambda_t*gradient
    x = y_t - gradient
    if i > 3:
        gamma_t = 2/i
        lambda_t = 6/i/(i+1)
    # print(x)
best = min(best, np.linalg.norm(A@x))
best_list2.append(best)
plt.loglog(np.arange(MAX_ITERS+1), best_list2, label="AGD lr=%s" % lr)
# plt.title("Gradient Descent lr=%s" % lr)
plt.xlabel("Iterations")
plt.ylabel("Value of Best Iterate")
leg = plt.legend(loc='upper center')
plt.show()
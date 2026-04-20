import numpy as np

X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) # Bipolar inputs
y = np.array([-1, -1, -1, 1]) # AND target

w1, w2, b = 0, 0, 0 # Blank slate

for i in range(len(X)):
    w1 += (X[i][0] * y[i]) # w_new = w_old + x*y
    w2 += (X[i][1] * y[i])
    b += y[i] # Update bias

print(w1, w2, b) # Final

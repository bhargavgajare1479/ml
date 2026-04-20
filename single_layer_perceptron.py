import numpy as np

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1]) # AND target
w, b, lr = np.array([0.0, 0.0]), 0.0, 0.1 # Init params
for epoch in range(10): # Training loop
    err_sum = 0
    for i in range(len(X)):
        pred = 1 if (np.dot(X[i], w) + b) >= 0 else 0 # Step function
        err = y[i] - pred # Calc error
        
        w += (lr * err * X[i]) # Update weight
        b += (lr * err) # Update bias
        err_sum += abs(err)
    
    if err_sum == 0: break # Stop if perfect

import numpy as np

def mcp(inputs, weights, threshold):
    return 1 if np.dot(inputs, weights) >= threshold else 0 # Threshold check

X = np.array([[0,0], [0,1], [1,0], [1,1]]) # Binary inputs

# AND
for i in X: print(mcp(i, [1, 1], 2)) # Needs both 1s
        
# OR
for i in X: print(mcp(i, [1, 1], 1)) # Needs at least one 1

# NAND
for i in X: print(mcp(i, [-1, -1], -1)) # Inhibitory logic

import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0]) # XOR targets

# 1 hidden layer (4 neurons), relu
model = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', max_iter=2000) 
model.fit(X, y) # Train

print(model.predict(X)) # Test


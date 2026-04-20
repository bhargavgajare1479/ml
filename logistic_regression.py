import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[1], [2], [3], [5], [6], [7]]) # 2D array
y = np.array([0, 0, 0, 1, 1, 1]) # Binary classes

model = LogisticRegression() # Init
model.fit(X, y) # Train

print(model.predict([[4.5]])) # Class prediction
print(model.predict_proba([[4.5]])) # Probabilities

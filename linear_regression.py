import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]]) # 2D array
y = np.array([10, 20, 30, 40, 50]) # 1D array

model = LinearRegression() # Init
model.fit(X, y) # Train

print(model.coef_, model.intercept_) # Weight, Bias
print(model.predict([[6]])) # Predict new

import numpy as np
from sklearn.svm import SVC
X = np.array([[1, 2], [2, 3], [2, 2], [8, 9], [9, 8], [9, 9]]) # 2D features
y = np.array([0, 0, 0, 1, 1, 1]) # Classes

model = SVC(kernel='linear') # Straight line
model.fit(X, y) # Train

print(model.predict([[7, 8]])) # Predict new
print(model.support_vectors_) # Check edge points


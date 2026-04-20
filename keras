import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[0,0], [0,1], [1,0], [1,1]]) # Inputs
y = np.array([[0], [1], [1], [0]]) # XOR target

model = Sequential() # Linear stack
model.add(Dense(8, input_dim=2, activation='relu')) # Hidden
model.add(Dense(1, activation='sigmoid')) # Output layer

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Setup
model.fit(X, y, epochs=500, verbose=0) # Train

print(np.round(model.predict(X))) # Predict

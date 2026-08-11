"""
Quadratic Neuron
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Quadratic coefficients
a, b, c = 1, -3, 2

# Generate training data
x = np.linspace(-10, 10, 200)
y = a * x**2 + b * x + c

# Reshape for neural network
x_train = x.reshape(-1, 1)
y_train = y.reshape(-1, 1)

# Neural Network
model = Sequential([
    Dense(40, activation='relu', input_shape=(1,)),
    Dense(40, activation='relu'),
    Dense(1)  # Output layer is linear
])

# Compile
model.compile(
    optimizer='adam',
    loss='mse'
)

# Train
model.fit(
    x_train,
    y_train,
    epochs=600,
    verbose=0
)

# Prediction
y_pred = model.predict(x_train, verbose=0)

# Plot
plt.plot(x, y, label='Actual Quadratic Equation')
plt.plot(x, y_pred, '--', label='NN Prediction')

plt.xlabel("x")
plt.ylabel("y")
plt.title("Neural Network Learning a Quadratic Function")
plt.legend()
plt.grid(True)
plt.show()
'''# Stochastic gradient descect
import numpy as np 

# Data Creation 
X = np.array([1,2,3,4,5], dtype=np.float64)
Y = 2 * X + 3

# Weight and bias initialization
w = 0.0
b = 0.0
lr = 0.01  #Learning data 

# Traing 
for epoch in range(0, 300):
    for x, y in zip(X, Y):
        pred = w * x + b

        # Derivatives of squared loss
        dw = -2 * x * (y - pred)
        db = -2 * (y - pred)

        # Update parameter
        w -= lr * dw
        b -= lr * db

print("Optional weight:", w)
print("Optional bias:", b)
predicted = w*X+b

# ploting
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.plot(X,Y, marker = "o", linewidth=2, label="Actual")
plt.plot(X, predicted, marker='s', linestyle='--', linewidth=2,label='pred')
plt.grid(True)
plt.legend()
plt.show()'''

# Stochastic gradient descect
import numpy as np 

# Data Creation 
X = np.array([1,2,3,4,5], dtype=np.float64)
Y = X ** 2

# Weight and bias initialization
w = 0.0
b = 0.0
lr = 0.01  #Learning data 

# Traing 
for epoch in range(0, 300):
    for x, y in zip(X, Y):
        pred = w * x + b

        # Derivatives of squared loss
        dw = -2 * x * (y - pred)
        db = -2 * (y - pred)

        # Update parameter
        w -= lr * dw
        b -= lr * db

print("Optional weight:", w)
print("Optional bias:", b)
predicted = w*X+b

# ploting
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.plot(X,Y, marker = "o", linewidth=2, label="Actual")
plt.plot(X, predicted, marker='s', linestyle='--', linewidth=2,label='pred')
plt.grid(True)
plt.legend()
plt.show()
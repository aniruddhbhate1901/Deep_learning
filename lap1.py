import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Physics parameters
u = 5 #initial velocity
a = 2 #acceleration 

#Time data
t = np.linspace(0,10,200)
s = u*t + 0.5 * a * (t ** 2)

# Reshape
t_train = t.reshape(-1,1)
s_train = s.reshape(-1,1)

# Neutral Network with TANH activation 
model = Sequential([
    Dense(40, activation='relu', input_shape=(1,)),
    Dense(40, activation='relu'),
    Dense(1) #output must be linear 
])

# Compile
model.compile(optimizer='adam', loss='mse') 

# Train
model.fit(t_train, s_train, epochs=600, verbose=0)

# pred
s_pred = model.predict(t_train)

# plot
plt.plot(t, s, label='Actual Physics Equation')
plt.plot(t, s_pred, "--", label="NN Prediction (Tanh)")
plt.xlabel("Time (t)")
plt.ylabel("Displacement (s)")
plt.title("Neutral Network Activation")
plt.legend()
plt.grid(True)
plt.show()
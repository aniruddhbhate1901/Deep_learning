import numpy as np 
X = np.array([1,2,3,4,5], dtype=float)
Y = np.array([5,7,9,11,13], dtype=float)

# Initialize
w=0
b=0

gw =0 
gb =0

lr=0.5
eps= 1e-8

#loop
for epoch in range(300):
    # pred
    pred = w*X+b
    dw = (-2/len(X))*np.sum(X*(Y-pred))
    db = (-2/len(X))*np.sum(Y-pred)

    gw += dw ** 2
    gb += db ** 2

    w -= lr*dw/(np.sqrt((gw)+eps))
    b -= lr*db/(np.sqrt((gb)+eps))

print(w,b)
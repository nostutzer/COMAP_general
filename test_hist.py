import numpy as np
import matplotlib.pyplot as plt


a = np.arange(10, 100)
print(a)

b = np.zeros(len(a)+2)
w = np.zeros_like(b)
b[0] = 0
b[-1] = 120 * 120
b[1:-1] = a

w[0] = w[-1] = 0
w[1:-1] = np.ones_like(a)
print(b)
print(w)

hist, edges, im = plt.hist(b, weights = w, bins = int(np.max(b) - np.min(b)))
print(hist)
print(edges)

hist = hist.reshape(120, 120)
print(hist.T)

fig, ax = plt.subplots()
ax.imshow(hist.T)
plt.show()
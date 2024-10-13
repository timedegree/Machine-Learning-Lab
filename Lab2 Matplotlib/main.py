import matplotlib.pyplot as plt
import numpy as np

xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
width, height = 2000, 2000
iter_times = 300

x, y = np.linspace(xmin, xmax, width), np.linspace(ymin, ymax, height) 
X, Y = np.meshgrid(x, y)
C = X + 1j * Y

Z = np.zeros(C.shape, dtype=complex)
iterations = np.zeros(C.shape, dtype=int)

for i in range(iter_times):
    mask = np.abs(Z) <= 2
    Z[mask] = Z[mask] ** 2 + C[mask]
    iterations[mask] = i  

plt.figure(figsize=(8, 8))
plt.imshow(iterations, cmap='inferno', extent=(xmin, xmax, ymin, ymax))
plt.colorbar(label='Iterations until divergence')
plt.title('Mandelbrot Set')
plt.show()




import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve
import random

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

length = 1000
N = 360

reel = []

# Spremenimo density in density0 na 2D namesto 3D matrike
density = np.zeros((N, N))
density0 = np.zeros((N, N))
density0[5, 5] = 0.9
density0[1, 1] = 0.5
density0[4, 9] = 0.7

# Vnaprej normalizirano Gaussovo jedro
g_size = 11
gauss = np.zeros((g_size, g_size))
for i in range(g_size):
    for j in range(g_size):
        dst = np.sqrt((i - g_size // 2) ** 2 + (j - g_size // 2) ** 2)
        gauss[i, j] = np.exp(-0.1 * dst ** 2)
gauss /= gauss.sum()  # Normaliziramo jedro

# Definiramo pravokotnik
def pravokotnik_g(x0, y0, x, y):
    for i in range(x0, x0 + x):
        for j in range(y0, y0 + y):
            density0[i, j] = (i - x0) / x  # Gradient

def pravokotnik(x0, y0, x, y):
    for i in range(x0, x0 + x):
        for j in range(y0, y0 + y):
            density0[i, j] = 1

pravokotnik_g(N // 2 - 40, N // 2 - 100, 80, 200)

reel.append(density0.copy())

# Uporabimo scipy konvolucijo za hitrejše računanje
for k in range(length):
    density = convolve(density0, gauss, mode='constant', cval=0.0)
    reel.append(density.copy())
    density0 = density.copy()

    pravokotnik(20,20,10,300)
    if k % (length // 10 + 1) == 0:
        print("Smo na ", (k / length) * 100, " %")

fig, ax = plt.subplots()
img = ax.imshow(reel[0], cmap='gray')

def update(frame):
    img.set_array(frame)
    return [img]

ani = FuncAnimation(fig, update, frames=reel, interval=1, blit=True)
plt.show()

# Shranjevanje končne slike
width1, height1 = N, N
im = Image.new('L', (width1, height1))
pix = im.load()
for y in range(height1):
    for x in range(width1):
        pix[x, y] = int(density[y, x] * 255.9999)

im = im.resize((width1 * 10, height1 * 10), resample=Image.NEAREST)
im.save('test.png')

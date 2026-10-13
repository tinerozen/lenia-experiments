import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve
import random

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

length = 50
N = 50

# Začetna mreža
density0 = np.zeros((N, N))
density0[5, 5] = 1
density0[1, 1] = 1
density0[4, 9] = 1
density0[40, 40] = 1
density0[41, 40] = 1
density0[42, 40] = 1
density0[42, 41] = 1
density0[41, 42] = 1

# Polje za definiranje naključnih celic
def rand_reg(x0, y0, x, y):
    for i in range(x0, x0 + x):
        for j in range(y0, y0 + y):
            density0[i, j] = random.choice([0, 1])

rand_reg(10, 10, 20, 20)

# Konvolucijsko jedro za sosednje celice
kernel = np.ones((3, 3), dtype=int)
kernel[1, 1] = 0  # Izključimo sredinsko celico

reel = [density0.copy()]

# Glavna zanka za iteracije
for k in range(length):
    density1 = convolve(density0, kernel, mode='constant', cval=0.0)
    density_next = np.where((density0 == 1) & ((density1 == 2) | (density1 == 3)), 1, 0)
    density_next = np.where((density0 == 0) & (density1 == 3), 1, density_next)
    reel.append(density_next)
    density0 = density_next.copy()

    # Prikaz napredka
    if k % (length // 10 + 1) == 0:
        print("Smo na ", (k / length) * 100, " %")

# Animacija
fig, ax = plt.subplots()
img = ax.imshow(reel[0], cmap='gray')

def update(frame):
    img.set_array(frame)
    return [img]

ani = FuncAnimation(fig, update, frames=reel, interval=100, blit=True)
plt.show()

# Shranjevanje končne slike
width1, height1 = N, N
im = Image.new('L', (width1, height1))
pix = im.load()
for y in range(height1):
    for x in range(width1):
        pix[x, y] = int(density0[y, x] * 255.9999)

im = im.resize((width1 * 10, height1 * 10), resample=Image.NEAREST)
im.save('test.png')

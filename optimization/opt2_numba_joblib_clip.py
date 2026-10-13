import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve
import random
from joblib import Parallel, delayed
from numba import jit

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

length = 800
N = 128
t = 0.1
randomnes = 0
global_variable = 1  # 1 -> on, 0 -> off
desired_mass = 2400
temp_x = desired_mass / (N * N)

reel = []

# Initialize density arrays
density = np.zeros((N, N))
density0 = np.zeros((N, N))
gradient = np.zeros((N, N))
density0[5, 5] = 0.9
density0[1, 1] = 0.5
density0[4, 9] = 0.7

# Precompute Gaussian kernel
g_size = 23
gauss = np.zeros((g_size, g_size))
for i in range(g_size):
    for j in range(g_size):
        dst = np.sqrt((i - g_size // 2) ** 2 + (j - g_size // 2) ** 2)
        gauss[i, j] = np.exp(-0.15 * (dst - 6) ** 2)
gauss /= gauss.sum()

kernel = np.ones((3, 3), dtype=float)
kernel2 = np.ones((27, 27), dtype=float)
kernel2[12, 12] = 0
kernel2[12, 13] = 0
kernel2[12, 14] = 0
kernel2[13, 12] = 0
kernel2[13, 13] = 0
kernel2[13, 14] = 0
kernel2[14, 12] = 0
kernel2[14, 13] = 0
kernel2[14, 14] = 0
kernel3 = np.ones((9, 9), dtype=float)  # Smaller kernel
kernel /= kernel.sum()
kernel2 /= kernel2.sum()

# Rectangle function to modify the density array
def pravokotnik(x0, y0, x, y):
    for i in range(x0, x0 + x):
        for j in range(y0, y0 + y):
            density0[i, j] = 1

def črni_pravokotnik(x0, y0, x, y):
    for i in range(x0, x0 + x):
        for j in range(y0, y0 + y):
            density0[i, j] = 0

def rand_reg(x0, y0, x, y):
    for i in range(x0, x0 + x):
        for j in range(y0, y0 + y):
            density0[i, j] = random.uniform(0, 1)

rand_reg(N // 2 - 40, N // 2 - 50, 60, 40)
rand_reg(N // 2 + 40 - 40, N // 2 + 50 - 40, 40, 40)

# Precompute the growth function
@jit(nopython=True)
def growth(x, r, temperature, temp_x):
    temperature_factor = temperature / temp_x
    temperature_factor = np.maximum(0.5, np.minimum(temperature_factor, 3))  # Clamping to ensure stability
    if global_variable == 1:
        return np.exp(-(340 * temperature_factor) * (x - 0.3 - r * randomnes) ** 2)
    return np.exp(-(340) * (x - 0.3 - r * randomnes) ** 2)

mass = []

# Clamping function
def clamp(n, min, max): 
    return np.clip(n, min, max)

flipper = False
temperature_starting_mass = density0.sum()

# Function to calculate the next density without using Numba for convolution
def calc_density_next(k, density0, gauss, t, global_variable, randomnes, temp_x):
    temperature = density0.sum() / (N * N)
    density1 = convolve(density0, gauss, mode='constant', cval=0.0)  # Convolution outside JIT
    rand = random.uniform(-0.01, 0.01)
    rand = clamp(rand, -0.2, 0.2)
    density_next = (density0 + t * (2 * growth(density1, rand, temperature, temp_x) - 1))
    for i in range(N):
        for j in range(N):
            density_next[i, j] = clamp(density_next[i, j], 0, 1)
    return density_next

# Parallelize the main loop using joblib
results = Parallel(n_jobs=-1)(delayed(calc_density_next)(k, density0, gauss, t, global_variable, randomnes, temp_x) for k in range(length))

# Create the animation
fig, ax = plt.subplots()
img = ax.imshow(results[0], cmap='gray')

def update(frame):
    img.set_array(frame)
    return [img]

ani = FuncAnimation(fig, update, frames=results, interval=30, blit=True)
plt.show()

# Plot the mass over time
plt.plot([np.sum(d) for d in results])
plt.show()

# Saving the final image
width1, height1 = N, N
im = Image.new('L', (width1, height1))
pix = im.load()
for y in range(height1):
    for x in range(width1):
        pix[x, y] = int(density0[y, x] * 255)

im = im.resize((width1 * 10, height1 * 10), resample=Image.NEAREST)
im.save('test.png')

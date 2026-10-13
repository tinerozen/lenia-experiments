from numba import jit
from joblib import Parallel, delayed
import numpy as np
import random
from scipy.ndimage import convolve
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


length = 800
N = 128
t = 0.1
randomnes = 0
global_variable = 1 # 1 -> on, 0 -> off
desired_mass = 2400
temp_x = desired_mass/(N*N)
reel = []


density = np.zeros((N, N))
density0 = np.zeros((N, N))
gradient = np.zeros((N, N))
density0[5, 5] = 0.9
density0[1, 1] = 0.5
density0[4, 9] = 0.7

g_size = 23
gauss = np.zeros((g_size, g_size))
for i in range(g_size):
    for j in range(g_size):
        dst = np.sqrt((i - g_size // 2) ** 2 + (j - g_size // 2) ** 2)
        gauss[i, j] = np.exp(-0.15 * (dst-6) ** 2)
gauss /= gauss.sum()  # Normaliziramo jedro



# Naredimo optimizacijo z Numba
@jit(nopython=True)
def growth(x, r, temperature, temp_x):
    temperature_factor = temperature / temp_x
    if temperature_factor < 0.5:
        temperature_factor = 0.5
    if temperature_factor > 3:
        temperature_factor = 3
    if global_variable == 1:
        return np.exp(-(340 * temperature_factor) * (x - 0.3 - r * randomnes) ** 2)
    if global_variable == 0:
        return np.exp(-(340) * (x - 0.3 - r * randomnes) ** 2)

@jit(nopython=True)
def clamp(n, min, max): 
    if n < min: 
        return min
    elif n > max: 
        return max
    else: 
        return n

@jit(nopython=True)
def rand_reg(x0, y0, x, y):
    for i in range(x0, x0 + x):
        for j in range(y0, y0 + y):
            density0[i, j] = random.uniform(0, 1)

# Dodela paralelno izvajanje
def calc_density_next(k, density0, gauss, t, global_variable, randnes, temp_x):
    temperature = density0.sum() / (N * N)
    density1 = convolve(density0, gauss, mode='constant', cval=0.0)
    rand = random.uniform(-0.01, 0.01)
    rand = clamp(rand, -0.2, 0.2)
    density_next = (density0 + t * (2 * growth(density1, rand, temperature, temp_x) - 1))
    
    for i in range(N):
        for j in range(N):
            density_next[i, j] = clamp(density_next[i, j], 0, 1)

    return density_next

# Uporaba paralelnega izvajanja za izračunanje vseh okvirjev
results = Parallel(n_jobs=-1)(delayed(calc_density_next)(k, density0, gauss, t, global_variable, randomnes, temp_x) for k in range(length))


fig, ax = plt.subplots()
img = ax.imshow(results[0], cmap='gray')

def update(frame):
    img.set_array(frame)
    return [img]

#x = np.linspace(1, len(mass))



ani = FuncAnimation(fig, update, frames=results, interval=30, blit=True)
plt.show()
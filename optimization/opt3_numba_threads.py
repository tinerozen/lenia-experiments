import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor

# Configuration constants
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
N = 128
LENGTH = 800
TIME_STEP = 0.1
DESIRED_MASS = 2400
TEMP_X = DESIRED_MASS / (N * N)
GAUSS_SIZE = 23
N_SQ = N * N
rand = 0.0
mass = []
# Initialize data structures
reel = []
density = np.zeros((N, N))
density0 = np.zeros((N, N))

# Precompute Gaussian kernel
x = np.arange(GAUSS_SIZE) - GAUSS_SIZE // 2
y = np.arange(GAUSS_SIZE) - GAUSS_SIZE // 2
xx, yy = np.meshgrid(x, y, indexing='ij')
dst = np.sqrt(xx**2 + yy**2)
gauss = np.exp(-0.15 * (dst - 6)**2)
gauss /= gauss.sum()

# Initialize kernels
kernel2 = np.ones((27, 27), dtype=float)
kernel2[12:15, 12:15] = 0
kernel2 /= kernel2.sum()

# Vectorized shape functions
def rand_reg(x0, y0, w, h):
    density0[x0:x0+w, y0:y0+h] = np.random.uniform(0, 1, (w, h))

def pravokotnik(x0, y0, w, h):
    density0[x0:x0+w, y0:y0+h] = 1

def crni_pravokotnik(x0, y0, w, h):
    density0[x0:x0+w, y0:y0+h] = 0

# Initial patterns
rand_reg(N//2-40, N//2-50, 60, 40)
rand_reg(N//2, N//2-40, 40, 40)
rand_reg(90, 90, 30, 30)
reel.append(density0.copy())

# Numba-optimized growth calculation
@njit(parallel=True, fastmath=True)
def numba_growth(density1, temp_factor, rand):
    return np.exp(-340 * temp_factor * (density1 - 0.3 - rand)**2)

# Numba-optimized density update
@njit(parallel=True)
def numba_update(density0, growth_vals, TIME_STEP):
    return np.clip(density0 + TIME_STEP * (2 * growth_vals - 1), 0, 1)

# Multithreaded convolution helper
def parallel_convolve(args):
    return convolve(*args)

# Modified main simulation loop
def run_simulation():
    pool = ThreadPoolExecutor()
    density0 = np.zeros((N, N))
    rand = 0.0
    density_buffer = np.empty_like(density0)
    
    for k in range(LENGTH):
        # ... [Keep pattern updates] ...
        
        # Asynchronous convolution
        conv_future = pool.submit(parallel_convolve, (density0, gauss, 'constant', 0.0))
        
        # Parallel growth calculations
        temperature = density0.sum() / N_SQ
        temp_factor = np.clip(temperature / TEMP_X, 0.5, 3)
        rand = np.clip(np.random.uniform(-0.01, 0.01) + rand, -0.2, 0.2)
        
        # Get convolution result
        density1 = conv_future.result()
        
        # Numba-accelerated calculations
        growth_vals = numba_growth(density1, temp_factor, rand)
        density_next = numba_update(density0, growth_vals, TIME_STEP)
    
        # Store results
        reel.append(density_next)
        mass.append(density_next.sum())
        density0 = density_next  # No copy needed as clip creates new array

        # Progress reporting
        if k % (LENGTH // 10 + 1) == 0:
            print(f"Progress: {k/LENGTH*100:.1f}%")

# Visualization
fig, ax = plt.subplots()
img = ax.imshow(reel[0], cmap='gray')

def update(frame):
    img.set_array(frame)
    return [img]

ani = FuncAnimation(fig, update, frames=reel, interval=30, blit=True)
plt.show()

plt.plot(mass)
plt.show()

import numpy as np
from numba import njit, prange
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve
import concurrent.futures

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

# Precompute Gaussian kernel
x = np.arange(GAUSS_SIZE) - GAUSS_SIZE // 2
y = np.arange(GAUSS_SIZE) - GAUSS_SIZE // 2
xx, yy = np.meshgrid(x, y, indexing='ij')
dst = np.sqrt(xx**2 + yy**2)
gauss = np.exp(-0.15 * (dst - 6)**2)
gauss /= gauss.sum()

@njit(nogil=True, cache=True)
def numba_growth(density1, temp_factor, rand):
    return np.exp(-340 * temp_factor * (density1 - 0.3 - rand)**2)

@njit(nogil=True, parallel=True, fastmath=True)
def numba_update(density0, growth_vals, TIME_STEP):
    result = np.empty_like(density0)
    for i in prange(density0.shape[0]):
        for j in prange(density0.shape[1]):
            result[i,j] = density0[i,j] + TIME_STEP * (2 * growth_vals[i,j] - 1)
            if result[i,j] < 0: result[i,j] = 0
            if result[i,j] > 1: result[i,j] = 1
    return result

class Simulation:
    def __init__(self):
        self.density0 = np.zeros((N, N))
        self.reel = []
        self.mass = []
        self.rand = 0.0
        self.pool = concurrent.futures.ThreadPoolExecutor()
        
        # Initialize patterns
        self.rand_reg(N//2-40, N//2-50, 60, 40)
        self.rand_reg(N//2, N//2-40, 40, 40)
        self.rand_reg(90, 90, 30, 30)
        self.reel.append(self.density0.copy())

    def rand_reg(self, x0, y0, w, h):
        self.density0[x0:x0+w, y0:y0+h] = np.random.uniform(0, 1, (w, h))

    def pravokotnik(self, x0, y0, w, h):
        self.density0[x0:x0+w, y0:y0+h] = 1

    def crni_pravokotnik(self, x0, y0, w, h):
        self.density0[x0:x0+w, y0:y0+h] = 0

    def run(self):
        for k in range(LENGTH):
            # # Update patterns
            # if k % 30 < 15:
            #     self.pravokotnik(60, 30, 10, 68)
            # else:
            #     self.crni_pravokotnik(60, 30, 10, 68)
            
            # if k == 120:
            #     self.rand_reg(N//2, N//2-40, 40, 40)

            # Async convolution
            future = self.pool.submit(convolve, self.density0, gauss, mode='constant', cval=0.0)
            
            # Calculate parameters
            temperature = self.density0.sum() / N_SQ
            temp_factor = np.clip(temperature / TEMP_X, 0.5, 3)
            self.rand = np.clip(np.random.uniform(-0.01, 0.01) + self.rand, -0.2, 0.2)
            
            # Get convolution result
            density1 = future.result()
            
            # Numba-accelerated calculations
            growth_vals = numba_growth(density1, temp_factor, self.rand)
            density_next = numba_update(self.density0, growth_vals, TIME_STEP)
            
            # Update state
            self.reel.append(density_next)
            self.mass.append(density_next.sum())
            self.density0 = density_next

            # Progress
            if k % (LENGTH // 10) == 0:
                print(f"Progress: {k/LENGTH*100:.1f}%")

    def visualize(self):
        fig, ax = plt.subplots()
        img = ax.imshow(self.reel[0], cmap='gray')
        
        def update(frame):
            img.set_array(frame)
            return [img]
        
        ani = FuncAnimation(fig, update, frames=self.reel, interval=30, blit=True)
        plt.show()
        
        plt.plot(self.mass)
        plt.show()
        
        im = Image.fromarray((self.density0 * 255).astype(np.uint8))
        im = im.resize((N*10, N*10), resample=Image.NEAREST)
        im.save('images/optimized_result.png')

if __name__ == "__main__":
    sim = Simulation()
    sim.run()
    sim.visualize()
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve
import random

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

length = 800
N = 128
t = 0.1
randomnes = 0
global_variable = 1 # 1 -> on, 0 -> off
desired_mass = 2400
temp_x = desired_mass/(N*N)

reel = []

# Spremenimo density in density0 na 2D namesto 3D matrike
density = np.zeros((N, N))
density0 = np.zeros((N, N))
gradient = np.zeros((N, N))
density0[5, 5] = 0.9
density0[1, 1] = 0.5
density0[4, 9] = 0.7

# Vnaprej normalizirano Gaussovo jedro
g_size = 23
gauss = np.zeros((g_size, g_size))
for i in range(g_size):
    for j in range(g_size):
        dst = np.sqrt((i - g_size // 2) ** 2 + (j - g_size // 2) ** 2)
        gauss[i, j] = np.exp(-0.15 * (dst-6) ** 2)
gauss /= gauss.sum()  # Normaliziramo jedro

#plt.imshow(gauss, cmap='gray')
#plt.show()



kernel = np.ones((3, 3), dtype=float)


kernel2 = np.ones((27, 27), dtype=float)
kernel2[12,12]=0
kernel2[12,13]=0 
kernel2[12,14]=0 
kernel2[13,12]=0 
kernel2[13,13]=0 
kernel2[13,14]=0 
kernel2[14,12]=0 
kernel2[14,13]=0
kernel2[14,14]=0

kernel3 = np.ones((9, 9), dtype=float)  # Uporabimo manjše jedro, npr. 9x9
kernel3 /= kernel3.sum()

kernel /= kernel.sum()
kernel2 /= kernel2.sum()

#plt.imshow(kernel2, cmap='gray')
#plt.show()



# Definiramo pravokotnik
def pravokotnik_g(x0, y0, x, y):
    for i in range(x0, x0 + x):
        for j in range(y0, y0 + y):
            density0[i, j] = (j - y0) / y  # Gradient

def pravokotnik(x0, y0, x, y):
    for i in range(x0, x0 + x):
        for j in range(y0, y0 + y):
            density0[i, j] = 1

def črni_pravokotnik(x0, y0, x, y):
    for i in range(x0, x0 + x):
        for j in range(y0, y0 + y):
            density0[i, j] = 0

def rand_reg(x0,y0, x, y):
    for i in range(x0,x0+x):
        for j in range(y0, y0+y):
            density0[i, j] = random.uniform(0,1)



rand_reg(N // 2 - 40, N // 2 - 50, 60, 40)
rand_reg(N // 2 + 40 -40, N // 2 + 50 -40, 40, 40)

rand_reg(90, 90, 30, 30)
#rand_reg(90, 31, 30, 30)
#rand_reg(31,90, 30, 30)

#rand_reg(N // 2 - 10, N // 2 - 128, 30, 30)

#rand_reg(N // 2 - 128, N // 2 - 128, 30, 20)

#rand_reg(N // 2 - 128, N // 2 - 10, 30, 20)

#pravokotnik(20,20,10,30)

reel.append(density0.copy())

def growth(x, r, temperature, temp_x):
    temperature_factor = temperature/temp_x
    if temperature_factor < 0.5:
        temperature_factor = 0.5
    if temperature_factor > 3:
        temperature_factor = 3
    if global_variable == 1:
        return np.exp(-(340*temperature_factor)*(x-0.3-r*randomnes)**2)
    if global_variable == 0:
        return np.exp(-(340)*(x-0.3-r*randomnes)**2)

# np.exp(-225*(x-0.3)**2) deluje vredu za gauss: gauss[i, j] = np.exp(-0.4 * (dst-3) ** 2)
# np.exp(-130*(x-0.35)**2) včasih nenehno raste, včasih pa umre, zdi se manj stabilna varjanta kot prejšnja

mass = []

def clamp(n, min, max): 
    if n < min: 
        return min
    elif n > max: 
        return max
    else: 
        return n
rand = 0
flipper = False
temperature_starting_mass = density0.sum()
# Uporabimo scipy konvolucijo za hitrejše računanje

for k in range(length):
    """
    density = convolve(density0, gauss, mode='constant', cval=0.0)
    reel.append(density.copy())
    density0 = density.copy()
    """
    if k % 30 == 0:
        flipper = True
    if (k - 15) % 30 == 0:
        flipper = False
    if flipper == True:
        pravokotnik(60,30, 10, 68)
    if flipper == False:
        črni_pravokotnik(60,30, 10, 68)

    if k == 120:
        rand_reg(N // 2 + 40 -40, N // 2 + 50 -40, 40, 40)

    temperature = density0.sum()/(N*N)

    density1 = convolve(density0, gauss, mode='constant', cval=0.0)
    #density_next = np.where((density2 >= 0.5) & ((density1 >= 0.07) & (density1 <= 0.90)), 1, 0)
    #density_next = np.where((density2 < 0.5) & ((density1 >= 0.10) & (density1 <= 0.90)), 1, 0)
    rand += random.uniform(-0.01,0.01)
    rand = clamp(rand, -0.2, 0.2)
    density_next = (density0 + t*(2*growth(density1, rand, temperature, temp_x)-1))
    for i in range(N):
        for j in range(N):
            density_next[i,j] = clamp(density_next[i,j], 0, 1)
    
    reel.append(density_next)
    mass.append(density_next.sum())
    density0 = density_next.copy()

    



    if k % (length // 10 + 1) == 0:
        print("Smo na ", (k / length) * 100, " %")

fig, ax = plt.subplots()
img = ax.imshow(reel[0], cmap='gray')

def update(frame):
    img.set_array(frame)
    return [img]

#x = np.linspace(1, len(mass))



ani = FuncAnimation(fig, update, frames=reel, interval=30, blit=True)
plt.show()


plt.plot(mass)
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

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import time

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True




lenght = 5
N = 10

reel = []

density = np.zeros((N,N,1))
density0 = np.zeros((N,N,1))
density0[5,5]=[1]
density0[1,1]=[0.5]
density0[4,9]=[0.7]

#plt.imshow(density0, cmap='gray')
#plt.show()

def disipacija(density0):
    

for k in range(lenght):
    for i in range(1,N-2):
        density[i,0,0] = np.divide((density0[i, 1, 0] + density0[i, 0, 0] + density0[i-1, 0, 0] + density0[i+1, 0, 0]), 5.0)
    for i in range(1,N-2):
        density[i,N-1,0] = np.divide((density0[i, N-2, 0] + density0[i, N-1, 0] + density0[i-1, N-1, 0] + density0[i+1, N-1, 0]), 5.0)

    for i in range(1,N-2):
        density[0,i,0] = np.divide((density0[1, i, 0] + density0[0, i, 0] + density0[0, i-1, 0] + density0[0, i+1, 0]), 5.0)
    for i in range(1,N-2):
        density[N-1,i,0] = np.divide((density0[N-2, i, 0] + density0[N-1, i, 0] + density0[N-1, i-1, 0] + density0[N-1, i+1, 0]), 5.0)

    for i in range(1,N-1):
        for j in range(1,N-1):
            density[i,j,0] = np.divide((density0[i-1, j, 0] + density0[i+1, j, 0] + density0[i, j-1, 0] + density0[i, j+1, 0] + density0[i, j, 0]), 5.0)

    #plt.imshow(density, cmap='gray')
    #plt.show()
    
    
    reel.append(density.copy())
    density0 = density.copy()

fig, ax = plt.subplots()
img = ax.imshow(reel[0], cmap='gray')

def update(frame):
    img.set_array(frame)
    return [img]

ani = FuncAnimation(fig, update, frames=reel, interval=500, blit=True)

plt.show()




width1 = N
height1 = N

im = Image.new('L', (width1, height1))
pix = im.load()
for y in range(height1):
    for x in range(width1):
        pix[x, y] = int(density[y][x] * 255.9999)

im = im.resize((width1 * 10, height1 * 10), resample=Image.NEAREST)
im.save('test.png')






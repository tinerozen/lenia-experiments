import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import time
import random

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True




lenght = 80
N = 360 # min 10

reel = []

density = np.zeros((N,N, 1))
density0 = np.zeros((N,N, 1))
density0[5,5]= 0.9
density0[1,1]=[0.5]
density0[4,9]=[0.7]

g_size = 11
g_div = 0
gauss = np.zeros((g_size, g_size, 1))
dst = 0
for i in range(g_size):
    for j in range(g_size):
        dst = np.sqrt((np.power(i - g_size//2, 2)+ np.power(j - g_size//2, 2)))
        gauss[i,j, 0] = np.exp(-0.1*np.power(dst,2))
        g_div += np.exp(-0.1*np.power(dst,2))

#print(g_div)


def krog(x,y,r):
    for i in range(100*int(2*np.pi)):
        density0[int((x + r * np.cos(i))),int((y + r * np.sin(i))), 0] = 1.0


#for i in range(200):
#    krog(500,600,i)

#plt.imshow(gauss, cmap='gray')
#plt.show()

def pravokotnik(x0,y0, x, y):
    for i in range(x0,x0+x):
        for j in range(y0, y0+y):
            density0[i, j, 0] = (i-x0)/x # <--- gradient

def rand_reg(x0,y0, x, y):
    for i in range(x0,x0+x):
        for j in range(y0, y0+y):
            density0[i, j, 0] = random.uniform(0,1)

pravokotnik(N//2-40, N//2-100, 80, 200)
#rand_reg(N//2-40, N//2-100, 80, 100)

reel.append(density0.copy())

for k in range(lenght):

    for i in range(20,N-20):
        for j in range(20,N-20):
            sum = 0
            for m in range(g_size):
                for n in range(g_size):
                    sum += density0[i-(g_size//2)+m, j-(g_size//2)+n, 0]*gauss[m,n,0]
            density[i,j,0] = sum/g_div
    """
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
    """
    #plt.imshow(density, cmap='gray')
    #plt.show()
    
    reel.append(density.copy())
    density0 = density.copy()
    if k % (lenght//10 + 1) == 0: print("Smo na ", (k/lenght)*100, " %")



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






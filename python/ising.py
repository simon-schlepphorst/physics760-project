#!/usr/bin/pyhton3
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import time


def init_spin_array(N):
    return np.random.choice((-1, 1), size=(N, N))


def find_neighbors(spin_array, lattice, x, y):
    left   = (x, y - 1)     
    right  = (x, (y + 1) % lattice)
    top    = (x - 1, y)
    bottom = ((x + 1) % lattice, y)

    return [spin_array[left[0], left[1]],
            spin_array[right[0], right[1]],
            spin_array[top[0], top[1]],
            spin_array[bottom[0], bottom[1]]]


def energy(spin_array, lattice, x ,y):
    return 2 * spin_array[x, y] * sum(find_neighbors(spin_array, lattice, x, y))



#Plot parameters
cmap = mpl.colors.ListedColormap(['black','white'])
bounds=[-1,0,1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


RELAX_SWEEPS = 50
lattice = input("Enter lattice size: ")
sweeps = input("Enter the number of Monte Carlo Sweeps: ")


if os.path.isdir('Images') is False:
    os.mkdir('Images')


#Systematic Sweeping (going pooint by point in order
def SS():
    for temperature in np.arange(0.1, 5.0, 0.1):
        if os.path.isdir('Images/T-'+str(temperature)) is True:
            continue
        if os.path.isdir('Images/T-'+str(temperature)) is False:
            os.mkdir('Images/T-'+str(temperature))
        spin_array = init_spin_array(lattice)
        mag = np.zeros(sweeps + RELAX_SWEEPS)
        for sweep in range(sweeps + RELAX_SWEEPS):
            for i in range(lattice):
                for j in range(lattice):
                    e = energy(spin_array, lattice, i, j)
                    if e <= 0:
                        spin_array[i, j] *= -1
                        continue
                    elif np.exp((-1.0 * e)/temperature) > random.random():
                        spin_array[i, j] *= -1
                        continue
                    plt.imsave('Images/T-'+str(temperature)+'/'+'step-'+str((i,j))+'.png',spin_array,format='png', cmap = cmap)
            mag[sweep] = abs(sum(sum(spin_array))) / (lattice ** 2)
        print(temperature, sum(mag[RELAX_SWEEPS:]) / sweeps)

#Random order Sweeping:
def RS():
    start = time.time()
    Temp = []
    Mag = []
    for temperature in np.arange(0.1, 5.0, 0.1):
        if os.path.isdir('Images/T-'+str(temperature)) is True:
            continue
        if os.path.isdir('Images/T-'+str(temperature)) is False:
            os.mkdir('Images/T-'+str(temperature))
        spin_array = init_spin_array(lattice)
        mag = np.zeros(sweeps + RELAX_SWEEPS)
        for sweep in range(sweeps + RELAX_SWEEPS):
            for i in range(sweeps):
                i = random.randint(0,lattice-1)
                j = random.randint(0,lattice-1)
                e = energy(spin_array, lattice, i, j)
                if e <= 0:
                    spin_array[i, j] *= -1
                    continue
                elif np.exp((-1.0 * e)/temperature) > random.random():
                    spin_array[i, j] *= -1
                    continue
                plt.imsave('Images/T-'+str(temperature)+'/'+'step-'+str(time.time()-start)+'.png',spin_array,format='png', cmap = cmap)
            mag[sweep] = abs(sum(sum(spin_array))) / (lattice ** 2)
        Temp =+ temperature
        Mag += sum(mag[RELAX_SWEEPS:]) / sweeps
        print(temperature, sum(mag[RELAX_SWEEPS:]) / sweeps)
    fig = plt.figure(1)
    plt.plot(Temp,Mag,'b-*',label='Data')
    plt.title('Magnetization vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Magnetization')
    fig.tight_layout()
    plt.show()

print "You may choose a random or systematic sweep by typing RS() or SS()"
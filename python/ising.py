#!/usr/bin/pyhton3
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib as mpl
import os


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

def n_step_pic(T,i,Arr,n):
    if i % n == 0:
        plt.imsave('Images/T-'+str(T)+'/'+'step-'+str(i/n).zfill(5)+'.png',Arr,format='png', cmap = cmap)
    else:
        return

def ACF(array,swep):
    C = np.zeros_like(array)
    for y,x in enumerate(array):
        for i in x:
            C[y,int(i)] = (array[y,0] * array[y,int(i)] - np.mean(array[y, :])**2)
    return C
    
def init_energy(spin_array, lattice):
    E = np.zeros_like(spin_array)
    for x in range(lattice):
        for y in range(lattice):
            E[x,y] = 2 * spin_array[x, y] * sum(find_neighbors(spin_array, lattice, x, y))
    return E.mean()

#Plot parameters
cmap = mpl.colors.ListedColormap(['black','white'])
bounds=[-1,0,1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


lattice = int(input("Enter lattice size: "))
sweeps = int(input("Enter the number of Monte Carlo Sweeps: "))
RELAX_SWEEPS = int(sweeps/100)
ACFE = np.zeros((50,sweeps + RELAX_SWEEPS))
ACFM = np.zeros((50,sweeps + RELAX_SWEEPS))


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
    T = []
    M = []
    steps = int(input("Enter how many steps in between images (set to 1 if every picture is wanted): "))
    for temperature in np.arange(0.1, 5.0, 0.1):
        if os.path.isdir('Images/T-'+str(temperature)) is True:
            continue
        if os.path.isdir('Images/T-'+str(temperature)) is False:
            os.mkdir('Images/T-'+str(temperature))
        spin_array = init_spin_array(lattice)
        E = init_energy(spin_array, lattice)
        mag = np.zeros(sweeps + RELAX_SWEEPS)
        for sweep in range(sweeps + RELAX_SWEEPS):
            
        #for i in range(sweeps):
            ii = random.randint(0,lattice-1)
            jj = random.randint(0,lattice-1)
            e = energy(spin_array, lattice, ii, jj)
            if e <= 0:
                spin_array[ii, jj] *= -1
                continue
            elif np.exp((-1.0 * e)/temperature) > random.random():
                spin_array[ii, jj] *= -1
                continue

            n_step_pic(temperature,sweep,spin_array,steps)
            
            mag[sweep] = abs(sum(sum(spin_array))) / (lattice ** 2)
            E = E + e
            ACFE[int(temperature*10 - 1),sweep] = E
            ACFM[int(temperature*10 - 1),sweep] = mag[sweep]
            
        T = T + [temperature]
        M = M + [sum(mag[RELAX_SWEEPS:]) / sweeps]
        print(temperature, sum(mag[RELAX_SWEEPS:]) / sweeps)
    
    c_e = ACF(ACFE,sweeps + RELAX_SWEEPS)
    c_m = ACF(ACFM,sweeps + RELAX_SWEEPS)
    
    print(T)
    print(M)
    print(ACFE)
    print(c_e)
    
    fig = plt.figure(1)
    plt.plot(T,M,'b-*',label='Data')
    plt.title('Magnetization vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Magnetization')
    fig.tight_layout()
    plt.show()
    
    fig = plt.figure()
    plt.plot(range(len(c_e[0])),c_e[0],'b-*',label='T = 0.1')
    plt.plot(range(len(c_e[0])),c_e[9],'r-o',label='T = 1.0')
    plt.plot(range(len(c_e[0])),c_e[19],'k-^',label='T = 2.0')
    plt.plot(range(len(c_e[0])),c_e[29],'c-s',label='T = 3.0')
    plt.plot(range(len(c_e[0])),c_e[39],'m-p',label='T = 4.0')
    plt.title('ACF of Energy')
    plt.xlabel('Time Step')
    plt.ylabel('ACF Value')
    fig.tight_layout()
    plt.legend(loc='best')
    plt.show()
    
    fig = plt.figure()
    plt.plot(range(len(c_m[0])),c_m[0],'b-*',label='T = 0.1')
    plt.plot(range(len(c_m[0])),c_m[9],'r-o',label='T = 1.0')
    plt.plot(range(len(c_m[0])),c_m[19],'k-^',label='T = 2.0')
    plt.plot(range(len(c_m[0])),c_m[29],'c-s',label='T = 3.0')
    plt.plot(range(len(c_m[0])),c_m[39],'m-p',label='T = 4.0')
    plt.title('ACF of Magnetization')
    plt.xlabel('Time Step')
    plt.ylabel('ACF Value')
    fig.tight_layout()
    plt.legend(loc='best')
    plt.show()
    
    #pl1 = ["ACFE","c_e","Temp","Mag"]
    #pl2 = [ACFE,c_e,T,M]
    #for i,j in zip(pl1,pl2):
    #    np.savetxt(i+".txt",j)

print("You may choose a random or systematic sweep by typing RS() or SS() \nBut I'm just gonna run RS()")

RS()

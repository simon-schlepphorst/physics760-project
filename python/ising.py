#!/usr/bin/pyhton3
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams.update({'font.size': 22})

def init_spin_array(N,choice):
    if choice == 'random':
        return np.random.choice((-1, 1), size=(N, N)).astype('int8')
    elif choice == 'hot':
        return np.resize([1,-1],(N,N))
    elif choice == 'cold':
        return np.ones((N,N), dtype='int8')

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

def ACC(x,k):
    n = int(len(x))
    k = int(k)
    xn = np.array(x)
    Eval = [i for i in (xn[:n-k]-xn.mean())*(xn[k:]-xn.mean())]
    Coeffs = 1/((n-k)*xn.var())
    results = Coeffs * sum(Eval)
    return results

def ACF(array,tstep):
    C = [[] for i in range(len(array))]
    for y,x in enumerate(array):
        C[y] = [ACC(x,i) for i in range(int(tstep))]
    return C

def MeanBlock(array,xran):
    RowLen = len(array[0])
    ColLen = len(array)
    Sigmas = []
    while RowLen%xran != 0:
        RowLen += -1
    for y in range(ColLen):
        SigList = []
        for B in range(1,xran):
            Array = [array[y][i:i+B] for i in range(0,RowLen,B)]
            if len(Array[0]) != len(Array[len(Array)-1]):
                Array = Array[0:len(Array)-2]
            Means = np.mean(Array,axis=1)
            SigmaMeans = np.std(Means)
            SigList.append(SigmaMeans)
        Sigmas.append(SigList)
    return Sigmas

def init_energy(spin_array, lattice):
    E = np.zeros_like(spin_array)
    for x in range(lattice):
        for y in range(lattice):
            E[x,y] = 2 * spin_array[x, y] * sum(find_neighbors(spin_array, lattice, x, y))
    return E.mean()

def init_mag(spin_array, lattice):
    return abs(sum(sum(spin_array))) / (lattice ** 2)

#Plot parameters
cmap = mpl.colors.ListedColormap(['black','white'])
bounds=[-1,0,1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


lattice = int(input("Enter lattice size [8]: ") or 8)
sweeps = int(input("Enter the number of Monte Carlo Sweeps [25000]: ") or 25000)
ACFTime = int(input("Enter the time for ACF to run over [500]: ") or 500)
choice = str(input("Choose array to start with (hot/cold/[random]): ") or "random")
RELAX_SWEEPS = int(sweeps/100)
Et = np.zeros((50,sweeps + RELAX_SWEEPS))
Mt = np.zeros((50,sweeps + RELAX_SWEEPS))


if os.path.isdir('Images') is False:
    os.mkdir('Images')


#Systematic Sweeping (going pooint by point in order
def SS():
    for temperature in np.arange(0.1, 5.0, 0.1):
        if os.path.isdir('Images/T-'+str(temperature)) is True:
            pass
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
    #steps = int(input("Enter how many steps in between images (set to 1 if every picture is wanted): "))
    for temperature in np.arange(0.1, 5.1, 0.1):
        if os.path.isdir('Images/T-'+str(temperature)) is True:
            pass
        if os.path.isdir('Images/T-'+str(temperature)) is False:
            os.mkdir('Images/T-'+str(temperature))
        spin_array = init_spin_array(lattice,choice)
        E = init_energy(spin_array, lattice)
        mag = np.zeros(sweeps + RELAX_SWEEPS)
        for sweep in range(sweeps + RELAX_SWEEPS):
            ii = random.randint(0,lattice-1)
            jj = random.randint(0,lattice-1)
            e = energy(spin_array, lattice, ii, jj)
            OrientI = spin_array[ii,jj]
            
            if e <= 0:
                spin_array[ii, jj] *= -1
                continue
            elif np.exp((-1.0 * e)/temperature) > random.random():
                spin_array[ii, jj] *= -1
                continue
            
            OrientF = spin_array[ii,jj]
            mag[sweep] = abs(sum(sum(spin_array))) / (lattice ** 2)

            if sweep == 0:
                Et[int(temperature*10 - 1)][0] = E
                  
            #n_step_pic(temperature,sweep,spin_array,steps)
            
            if OrientF == OrientI and sweep != 0:
                Et[int(temperature*10 - 1)][sweep] = Et[int(temperature*10 - 1)][sweep-1]
            if OrientF != OrientI and sweep != 0:
                Et[int(temperature*10 - 1)][sweep] = Et[int(temperature*10 - 1)][sweep-1]+e
            
            Mt[int(temperature*10 - 1),sweep] = mag[sweep]
            
        print("Temp ",[temperature], "and Mag ",[sum(mag[RELAX_SWEEPS:]) / sweeps]," Appending...\n")    
        T.append(temperature)
        M.append(sum(mag[RELAX_SWEEPS:]) / sweeps)
        
#    print("Getting ACF Function...\n")
#    c_e = ACF(Et,ACFTime)
#    c_m = ACF(Mt,ACFTime)
#    
#    print("ACF Function Complete\n")
    
    print("Finding Errors via Blocking\n")
    
    xRange = [i for i in range(1,500)]
    Sigmas = [MeanBlock(Et,500),MeanBlock(Mt,500)]
    
    fig = plt.figure(4)
    plt.plot(xRange,Sigmas[0][0],'b-*',label='T = 0.1')
    plt.plot(xRange,Sigmas[0][9],'r-o',label='T = 1.0')
    plt.plot(xRange,Sigmas[0][19],'k-^',label='T = 2.0')
    plt.plot(xRange,Sigmas[0][29],'c-s',label='T = 3.0')
    plt.plot(xRange,Sigmas[0][39],'m-p',label='T = 4.0')
    plt.plot(xRange,Sigmas[0][49],'g-h',label='T = 5.0')
    plt.title('Error of the Energy vs Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('$\sigma$')
    plt.xlim(0,len(xRange))
    fig.tight_layout()
    plt.legend(loc='best')
    plt.show()
    
    fig = plt.figure(5)
    plt.plot(xRange,Sigmas[1][0],'b-*',label='T = 0.1')
    plt.plot(xRange,Sigmas[1][9],'r-o',label='T = 1.0')
    plt.plot(xRange,Sigmas[1][19],'k-^',label='T = 2.0')
    plt.plot(xRange,Sigmas[1][29],'c-s',label='T = 3.0')
    plt.plot(xRange,Sigmas[1][39],'m-p',label='T = 4.0')
    plt.plot(xRange,Sigmas[1][49],'g-h',label='T = 5.0')
    plt.title('Error of the Magnetization vs Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('$\sigma$')
    plt.xlim(0,len(xRange))
    fig.tight_layout()
    plt.legend(loc='best')
    plt.show()
    
#    for i in range(50):
#        plt.plot(xRange,Sigmas[1][i],label='T = ' + str((1+i)/10))
#    plt.legend(loc='best',prop={'size':8})
    
    fig = plt.figure(1)
    plt.errorbar(T,M,yerr=np.sqrt(np.var(M)/sweeps),fmt='b-*',label='Data')
    plt.title('Magnetization vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Magnetization')
    fig.tight_layout()
    plt.draw()
    plt.pause(1)
    input("Hit ENTER ...")
    plt.close()
    
    '''
    Since we now have the blocking, it would be useful to recall the array arrangements here
    As you can see, we now need to apply the blocking to the M matrix so we can apply the coorect
    error. So here we can recalculate all the magnetizations blocked and then append their means
    to the M matrix. 
    '''
#   Naive blocked errors as it has just picked the 100th step which seems ok for the most part.
    
    BlockedSigmas = np.array(Sigmas)[1,:,45]
    
    Sigmas = np.array(Sigmas)
    Positions = []
    PercentDifference = .0025
    for t in range(len(Sigmas[1])):
        for i in range(len(Sigmas[1][0])):
            if Sigmas[1][t][i] >= Sigmas[1][t][i-1]*(1-PercentDifference) and\
                 Sigmas[1][t][i] <= Sigmas[1][t][i-1]*(1+PercentDifference) and\
                 Sigmas[1][t][i] >= Sigmas[1][t][i+1]*(1-PercentDifference) and\
                 Sigmas[1][t][i] <= Sigmas[1][t][i+1]*(1+PercentDifference):
                     Positions.append(i)
                     break

    BlockedSigmas2 = [Sigmas[1,i,j] for i,j in enumerate(Positions)]
    
    fig = plt.figure(6)
    plt.errorbar(T,M,yerr=BlockedSigmas,fmt='b-o',label='Naiive Blocking Choice')
    plt.errorbar(T,M,yerr=BlockedSigmas2,fmt='r-o',label='Individual Blocking Choice')
    plt.title('Magnetization vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Magnetization')
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()
    
#    fig = plt.figure(2)
#    plt.plot(range(len(c_e[0])),c_e[0],'b-*',label='T = 0.1')
#    plt.plot(range(len(c_e[0])),c_e[9],'r-o',label='T = 1.0')
#    plt.plot(range(len(c_e[0])),c_e[19],'k-^',label='T = 2.0')
#    plt.plot(range(len(c_e[0])),c_e[29],'c-s',label='T = 3.0')
#    plt.plot(range(len(c_e[0])),c_e[39],'m-p',label='T = 4.0')
#    plt.title('ACF of Energy')
#    plt.xlabel('Time Step')
#    plt.ylabel('ACF Value')
#    plt.xlim(0,len(c_e[0]))
#    fig.tight_layout()
#    plt.legend(loc='best')
#    plt.show()
    
    #fig = plt.figure(3)
    #plt.plot(range(len(c_m[0])),c_m[0],'b-*',label='T = 0.1')
    #plt.plot(range(len(c_m[0])),c_m[9],'r-o',label='T = 1.0')
    #plt.plot(range(len(c_m[0])),c_m[19],'k-^',label='T = 2.0')
    #plt.plot(range(len(c_m[0])),c_m[29],'c-s',label='T = 3.0')
    #plt.plot(range(len(c_m[0])),c_m[39],'m-p',label='T = 4.0')
    #plt.title('ACF of Magnetization')
    #plt.xlabel('Time Step')
    #plt.ylabel('ACF Value')
    #plt.xlim(0,len(c_m[0]))
    #fig.tight_layout()
    #plt.legend(loc='best')
    #plt.show()


    

print("You may choose a random or systematic sweep by typing RS() or SS() \nBut I'm just gonna run RS()")

RS()

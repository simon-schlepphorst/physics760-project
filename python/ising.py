#!/usr/bin/python3
import itertools
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import sys
import tqdm

import configparser

###############################################################################
#           Lattice generation and simple functions                           #
###############################################################################

def init_spin_lattice(N, start='random'):
    '''
    Fills a array of shape N with random choice of 1 and -1

    :praram tuple N: shape of desired lattice
    :returns np.array: ising lattice
    '''
    if start == 'random':
        return np.random.choice([-1,1], N).astype(np.int8)
    elif start == 'hot':
        return np.resize([1,-1], N).astype(np.int8)
    elif start == 'cold':
        return np.ones(N).astype(np.int8)

#FIXME delete if not needed anymore
def init_spin_array(N,choice):
    if choice == 'random':
        return np.random.choice((-1, 1), size=(N, N)).astype('int8')
    elif choice == 'hot':
        return np.resize([1,-1],(N,N))
    elif choice == 'cold':
        return np.ones((N,N), dtype='int8')

def neighbors(lattice, point):
    '''
    Returns a iterator of neighboars of point.

    :param np.array lattice: numpy array of same number of dimensions as point
    :param tuple point: index vector
    :returns iter(tuple(int)): Neighbor iterator
    '''
    assert len(lattice.shape) == len(point)

    d = len(point)

    for k in range(d):
       s = np.zeros_like(point)
       s[k] = 1
       yield tuple((point + s) % lattice.shape[k])
       yield tuple((point - s) % lattice.shape[k])


def energy_simple(lattice, j=1):
    '''
    Calculates the energy on a lattice

    :param np.array lattice: numpy array with spins
    :returns np.array: energies on the lattice points
    '''

    d = len(lattice.shape)

    return np.sum( -j * lattice / 2 * np.sum(
            np.roll(lattice, shift, axis)
            for shift in [-1, 1]
            for axis in range(d)))


def energy_change(lattice, point, j=1):
    '''
    Returns the change in Energie with a spefic flipped value

    :param np.array lattice: numpy array with spins
    :param tuple flipped: index vector of spin to flip
    :returns int: Energy change
    '''
    return 2 * j * lattice[point] * np.sum(
            lattice[i] for i in neighbors(lattice, point))

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
    if os.path.isdir('Images') is False:
        os.mkdir('Images')
    if i % n == 0:
        plt.imsave('Images/T-'+str(T)+'/'+'step-'+str(i/n).zfill(5)+'.png',Arr,format='png', cmap = cmap)
    else:
        return

#FIXME --->
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
# <--- replace with statsmodels.tsa.stattools.acf

def MeanBlock(array,xran):
    '''
    :param np.array array: array with magnetisations for each step
    :param int xran: maximal blocking size
    :returns list Sigmas: containing the calcuated deviations per blocking size
    '''
    #FIXME was meant for an 50xSweeps(T) array
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

#FIXME marked for deletion, need to resolve dependencies
def init_energy(spin_array, lattice):
    E = np.zeros_like(spin_array)
    for x in range(lattice):
        for y in range(lattice):
            E[x,y] = - 1/2 * spin_array[x, y] * sum(find_neighbors(spin_array, lattice, x, y))
    return np.sum(E)

def init_mag(spin_array, lattice):
    return abs(sum(sum(spin_array))) / (lattice ** 2)

def make_cluster(spin_array, lattice, N, temperature):
    '''
    This function makes one cluster out of a starting point. 
    That is to say it outputs a [list] with all the points in the cluster    
    '''
    assert len(N) == 2
    x, y = N

    Origin = [x,y]
    Cluster = [(x,y)]
    i = 1
    while True:
        neighbors = find_neighbors(spin_array, lattice,Origin[0],Origin[1])
        L,R,T,B = neighbors[0],neighbors[1],neighbors[2],neighbors[3]
        try:
            if i >= 5 and i >= len(Cluster) and Origin == [Cluster[-2][0],Cluster[-2][1]]:
                break
        except IndexError:
            break
        while True:
            OriginalSpin = spin_array[Origin[0], Origin[1]]
            if OriginalSpin == L and 1. - np.exp(-2.0/temperature) > random.random() and \
                                                ((Origin[0], Origin[1] - 1) not in Cluster):
                Cluster.append((Origin[0], Origin[1] - 1))
            if OriginalSpin == R and 1. - np.exp(-2.0/temperature) > random.random() and \
                                                ((Origin[0], (Origin[1] + 1) % lattice)):
                Cluster.append((Origin[0], (Origin[1] + 1) % lattice) not in Cluster)
            if OriginalSpin == T and 1. - np.exp(-2.0/temperature) > random.random() and \
                                                ((Origin[0]-1, Origin[1]) not in Cluster):
                Cluster.append((Origin[0]-1, Origin[1]))
            if OriginalSpin == B and 1. - np.exp(-2.0/temperature) > random.random() and \
                                                (((Origin[0] + 1) % lattice, Origin[1]) not in Cluster):
                Cluster.append(((Origin[0] + 1) % lattice, Origin[1]))
            try:
                Origin = [Cluster[i][0],Cluster[i][1]]
            except IndexError:
                Origin = [Cluster[-2][0],Cluster[-2][1]]
            i +=1
            break
    return Cluster

def cluster_merge(lists):
    '''
    Parameter lists: list of lists. List of clusters as lists. eg:
        [cluster1, cluster2, ...]
    Outputs modlist: lists of sets. List of clusters as sets.
    '''
    modlist = []
    for i in range(len(lists)):
        modlist.append(set(lists[i]))
    while True:
        for set1,set2 in itertools.combinations(modlist,2):
            try:
                index1 = modlist.index(set1)
            except ValueError:
                break
            print(set1,set2,index1)
            if not set1.isdisjoint(set2):
                modlist[index1] = set1.union(set2)
                modlist.remove(set2)
        else:
            break
    return modlist

def cluster_energy_change(lattice, clist, j=1):
    '''
    Param np.array lattice: numpy array with spins
    Param list of 2-tuples clists: points that are members of a cluster
    outputs energy of the entire cluster with respect to its neighbors
    Note: This does "double count" neighbors because one point may influence
        more than one point within the cluster.     
    '''
    E = []
    for i in range(len(clist)):
        point = clist[i]
        neighborlist = list(neighbors(lattice, point))
        for jloop in neighborlist:
            if jloop in clist:
                neighborlist.remove(jloop)
            else:
                pass
        E.append(2 * j * lattice[point] * np.sum(lattice[i] for i in neighborlist))
    return sum(E)

###############################################################################
#           Image saving parameters                                           #
###############################################################################

mpl.rcParams.update({'font.size': 22})
cmap = mpl.colors.ListedColormap(['black','white'])
bounds=[-1,0,1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

###############################################################################
#           Run simulations                                                   #
###############################################################################

def RS(parameters):
    ''' Random order sweeping
    :param dict parameters: dictionary with parameters
    '''

    sweeps = parameters['mc_sweeps']
    J = parameters['lattice_interaction']

    #creaty list of lattice and initialize first one
    lat_list = np.array([np.zeros(parameters['lattice_size'], dtype=np.int8) for _ in range(sweeps)])
    lat_list[0] = init_spin_lattice(parameters['lattice_size'], parameters['lattice_init'])

    #initialize list with easy to spot values
    T = np.array([parameters['mc_temp']]*sweeps)
    E = np.array([np.nan]*sweeps)
    M = np.array([np.nan]*sweeps)
    A = np.array([np.nan]*sweeps)

    E[0] = energy_simple(lat_list[0], J)
    M[0] = np.sum(lat_list[0])

    with tqdm.tqdm(desc= 'T='+str(T[0]), total=sweeps,  dynamic_ncols=True) as bar:
        for sweep in range(1, sweeps):
            bar.update()
            acceptance = 0
            lat_list[sweep] = lat_list[sweep - 1]
            E[sweep] = E[sweep - 1]
            M[sweep] = M[sweep - 1]

            for _ in range(len(lat_list[0].flatten())):
                # if the lattice has a strange size point will still be inside
                point = []
                for i in range(len(lat_list[0].shape)):
                    point.append(np.random.randint(0, lat_list[0].shape[i]))
                point = tuple(point)

                e = energy_change(lat_list[sweep], point, J)
                accept = (np.random.random() < np.exp(-np.divide(e, T[sweep - 1]))) or (e <= 0)

                if accept:
                    acceptance += 1
                    lat_list[sweep][point] *= -1
                    E[sweep] += e
                    M[sweep] += 2*lat_list[sweep][point]
                else:
                    pass

            A[sweep] = acceptance / len(lat_list[0].flatten())
        bar.update()

    with tqdm.tqdm(desc='Saving ...', total=1, dynamic_ncols=True) as bar:
        if parameters['save_lat']:
            np.savez_compressed(os.path.join(parameters['dirname'], "simulation.npz"), lat=lat_list, T=T, E=E, M=M, A=A)
        bar.update()

def CS(parameters):
    ''' Swendsen-Wang Cluster algorithm

    :saves lattice, ..., cluster list:
    '''
    raise NotImplementedError

    with tqdm.tqdm(desc='Saving ...', total=1, dynamic_ncols=True) as bar:
        if parameters['save_lat']:
            np.savez_compressed(os.path.join(parameters['dirname'], "simulation.npz"), lat=lat_list, T=T, E=E, M=M, A=A)
        bar.update()


def load_config(dirname, parameters):
    '''loads and parses a given config file

    :param filename: filename
    :param dict parameters: dictionary to fill values in
    '''
    config = configparser.ConfigParser()
    with open(os.path.join(dirname, "config.ini")) as f:
        config.read_file(f)

        #read options
        try:
            parameters['dirname'] = dirname
            parameters['lattice_size'] = eval(config['lattice']['size'])
            parameters['lattice_init'] = config['lattice'].get('init')
            if not parameters['lattice_init'] in ('hot', 'cold', 'random'):
                raise ValueError(parameters['lattice_init'])
            #currently not in use
            parameters['lattice_interaction'] = config['lattice'].getint('interaction')

            parameters['mc_sweeps'] = config['markov chain'].getint('sweeps')
            parameters['mc_start'] = config['markov chain'].getint('start')
            parameters['mc_temp'] = config.getfloat('markov chain', 'temperature')
            parameters['mc_algorithm'] = config.get('markov chain', 'algorithm')
            if not parameters['mc_algorithm'] in ('Monte Carlo', 'Cluster'):
                raise ValueError(parameters['mc_algorithm'])

            parameters['save_vol'] = config.getint('save', 'volume')
            parameters['save_pic'] = config.getboolean('save', 'pictures')
            parameters['save_lat'] = config.getboolean('save', 'lattice')
        except:
            print("Ooops. Some config is rotten in the state of Denmark.")
            raise

def run_sim(dirname):
    '''run simulation from config
    :param str dirname: directory to load the config from
    :output file: saves numpy arrays in given directory
    '''

    parameters = {} #dictionary to store the values of the config file
    load_config(dirname, parameters)


    def SS(parameters):
        ''' Systematic Sweeping (going pooint by point in order)
        :param dict parameters: dictionary with paramters
        '''
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



    if (parameters['mc_algorithm'] == 'Monte Carlo'):
        RS(parameters)
    elif (parameters['mc_algorithm'] == 'Cluster'):
        CS(parameters)

###############################################################################
#           Crunch the numbers                                                #
###############################################################################
    #TODO Calculation of the results should happen in extra function
    #     so reading runs from file is supported
def load_sim(dirname):
    '''load simulation
    :param str dirname: directory to load the simulation from
    :return dict dictionary: dictionary containing the results for given simualtion
    '''
    parameters = {} #dictionary to store the values of the config file

    # Read values from config if exist
    if os.path.isfile("config.ini"):
        load_config("config.ini", parameters)
    else:
        raise FileNotFoundError

    #translate options for legacy reasons --->
    lattice_N = parameters['lattice_size']
    lattice_state = parameters['lattice_state']
    lattice_J = parameters['lattice_interaction']
    mc_temp = parameters['mc_temp']
    mc_sweeps = parameters['mc_sweeps']
    mc_alg = parameters['mc_algorithm']
    save_vol = parameters['save_vol']
    save_pic = parameters['save_pic']
    save_lat = parameters['save_lat']

    lattice = parameters['lattice_size'][0] #TODO prepare rest of code for tuples
    sweeps = parameters['mc_sweeps']
    ACFTime = 500
    choice = parameters['lattice_state']

    RELAX_SWEEPS = int(sweeps/100)
    Et = np.zeros((50,sweeps + RELAX_SWEEPS))
    Mt = np.zeros((50,sweeps + RELAX_SWEEPS))
    #<--- FIXME

    # Read values from save if exist
    if os.path.isfile("save.npz"):
        try:
            f_npz = np.load('save.npz')
            lat_list = f_npz['lat']
            T=f_npz['T']
            E=f_npz['E']
            M=f_npz['M']
        except:
            raise
    else:
        raise FileNotFoundError


    #FIXME Glue Code to get the names right
    raise NotImplementedError


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
    plt.show()
    
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


###############################################################################
#           Execute some stuff if directly called                             #
###############################################################################

if __name__ == "__main__":

    if (len(sys.argv) == 2) and (os.path.isfile(sys.argv[1])):
        dirname = os.path.dirname(sys.argv[1])
        run_sim(dirname)
    else:
        print("Stupid you, didn't gave me a config.ini")


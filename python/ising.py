#!/usr/bin/python3
import itertools
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import os
import sys
import tqdm

import IPython

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

def MeanBlock(array, limit):
    '''
    :param np.array array: array with magnetisations for each step
    :param int limit: maximal blocking size
    :returns np.array Sigmas: containing the calcuated deviations per blocking size
    '''
    Sigmas = []

    for blocksize in range(1, limit):

        last_index = len(array) - len(array) % blocksize
        if last_index <= 0:
            Sigmas.append(np.nan)
            continue
        Array = np.array([array[i:i+blocksize] for i in range(0,last_index, blocksize)])
        Sigmas.append(np.mean( np.var(Array, axis=1)))

    return np.array(Sigmas)

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

#mpl.rcParams.update({'font.size': 22})
#cmap = mpl.colors.ListedColormap(['black','white'])
#bounds=[-1,0,1]
#norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

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
    :param dict parameters: dictionary with parameters
    :saves lattice, T, E, M, A, cluster:
    '''
    sweeps = parameters['mc_sweeps']
    J = parameters['lattice_interaction']

    #creaty list of lattice and initialize first one
    lat_list = np.array([np.zeros(parameters['lattice_size'], dtype=np.int8) for _ in range(sweeps)])
    lat_list[0] = init_spin_lattice(parameters['lattice_size'], parameters['lattice_init'])

    lat_bond = np.array([np.zeros(parameters['lattice_size'], dtype=np.int) for _ in range(sweeps)])

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
            accept_prop = 1 - np.exp(- 2.0 * J / T[sweep])

            lat_list[sweep] = lat_list[sweep - 1]
            bonds = []

            # Iterate over lattice
            it = np.nditer(lat_list[sweep], flags=['multi_index'])
            while not it.finished:

                point = it.multi_index
                for neighbor in neighbors(lat_list[sweep], point):
                    if lat_list[sweep][point] == lat_list[sweep][neighbor] and \
                            np.random.random() < accept_prop:
                        points = [point, neighbor]
                        isbond = [i for i, j in enumerate(bonds) if any(x in j for x in points)]
                        if isbond:
                            # merge clusters
                            first, *rest = [j for i, j in enumerate(bonds) if i in isbond]
                            rest.append(points)

                            # append elements from lists in rest
                            for i in rest:
                                first.extend(j for j in i)

                            first = list(set(first))
                            bonds = [j for i,j in enumerate(bonds) if i not in isbond] + [first]

                        else:
                            bonds = bonds + [points]

                    else:
                        pass

                it.iternext()

            #update spins on lattice
            for index, cluster in enumerate(bonds, start=1):
                new_value = np.random.choice([-1,1]).astype(np.int8)
                for point in cluster:
                    lat_list[sweep][point] = new_value
                    lat_bond[sweep][point] = index

            E[sweep] = energy_simple(lat_list[sweep])
            M[sweep] = np.sum(lat_list[sweep])
        bar.update()

    with tqdm.tqdm(desc='Saving ...', total=1, dynamic_ncols=True) as bar:
        if parameters['save_lat']:
            np.savez_compressed(os.path.join(parameters['dirname'], "simulation.npz"), lat=lat_list, T=T, E=E, M=M, A=A, cluster=lat_bond)
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
def load_sim(dirname, Plot=False):
    '''load simulation
    :param str dirname: directory to load the simulation from
    :return dict dictionary: dictionary containing the results for given simualtion
    '''
    parameters = {} #dictionary to store the values of the config file
    load_config(dirname, parameters)

    #translate options for legacy reasons --->
    lattice_N = parameters['lattice_size']
    lattice_state = parameters['lattice_init']
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
    choice = parameters['lattice_init']

    RELAX_SWEEPS = int(sweeps/100)
    Et = np.zeros((50,sweeps + RELAX_SWEEPS))
    Mt = np.zeros((50,sweeps + RELAX_SWEEPS))
    #<--- FIXME

    # Read values from save
    with np.load(os.path.join(parameters['dirname'], 'simulation.npz')) as data:
        lat_list = data['lat']
        T = data['T']
        # normalize on lattice size
        E = data['E'] / len(lat_list[0].flatten())
        M = data['M'] / len(lat_list[0].flatten())
        A = data['A']
        if parameters['mc_algorithm'] == 'Cluster':
            lat_bond = data['cluster']

    # setting relax sweeps to thermalise first
    if parameters['mc_algorithm'] == 'Monte Carlo':
        parameters['relax_sweeps'] = 1000
    elif parameters['mc_algorithm'] == 'Cluster':
        parameters['relax_sweeps'] = 200
    else:
        raise NotImplementedError


    # make one big plot with subplots
    if Plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
        plt.suptitle('Temperature, $k_b T={:1.2f}$, starting {}'.format(parameters['mc_temp'], parameters['lattice_init']))

    # Plot E, M against time
    if Plot:
        ax1.set_title('E, M vs Time')
        ax1.set_xlabel('Sweep')
        ax1.set_ylabel('#')
        ax1.plot(np.arange(len(E)), E, 'r-o', label='Energy')
        ax1.plot(np.arange(len(M)), M, 'b-*', label='Magnetisation')
        ax1.legend(loc='best')

    # calculate errors with blocking
    max_blocksize = (parameters['mc_sweeps'] - parameters['relax_sweeps']) // 2

    E_sigma_b = np.sqrt(MeanBlock(E[parameters['relax_sweeps']:], max_blocksize))
    M_sigma_b = np.sqrt(MeanBlock(M[parameters['relax_sweeps']:], max_blocksize))
    blocks = np.arange(len(E_sigma_b))

    # Plot blocksize vs sigma
    if Plot:
        ax2.set_title('Error vs Block Size')
        ax2.set_xlabel('Block Size')
        ax2.set_ylabel('$\sigma$')
        ax2.plot(blocks, E_sigma_b, 'r-o', label='Energy')
        ax2.plot(blocks, M_sigma_b, 'b-*', label='Magnetisation')
        ax2.legend(loc='best')

    if Plot:
        fig.tight_layout()
        plt.show()


    # return values for comparisation
    return parameters, E_sigma_b, M_sigma_b, E, M, A

    '''
    print("Finding Errors via Blocking\n")
    
    Sigmas = [MeanBlock(E[100:],500),MeanBlock(M[100:],500)]
    xRange = [i+1 for i in range(len(Sigmas[0]))]

    mask=np.isfinite(Sigmas)
    
    fig = plt.figure(4)
    plt.plot(xRange,Sigmas[0],'b-*',label='T = {:1f}'.format(parameters['mc_temp']))
    #plt.plot(xRange,Sigmas[0],'r-o',label='T = 1.0')
    #plt.plot(xRange,Sigmas[0],'k-^',label='T = 2.0')
    #plt.plot(xRange,Sigmas[0],'c-s',label='T = 3.0')
    #plt.plot(xRange,Sigmas[0],'m-p',label='T = 4.0')
    #plt.plot(xRange,Sigmas[0],'g-h',label='T = 5.0')
    plt.title('Error of the Energy vs Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('$\sigma$')
    plt.xlim(0,len(xRange))
    fig.tight_layout()
    plt.legend(loc='best')
    plt.show()
    
    fig = plt.figure(5)
    #plt.plot(xRange[mask[1]],Sigmas[1][mask[1]],'b-*',label='T = 0.1')
    plt.plot(xRange,Sigmas[1],'b-*',label='T = {:1f}'.format(parameters['mc_temp']))
    #plt.plot(xRange,Sigmas[1],'r-o',label='T = 1.0')
    #plt.plot(xRange,Sigmas[1],'k-^',label='T = 2.0')
    #plt.plot(xRange,Sigmas[1],'c-s',label='T = 3.0')
    #plt.plot(xRange,Sigmas[1],'m-p',label='T = 4.0')
    #plt.plot(xRange,Sigmas[1],'g-h',label='T = 5.0')
    plt.title('Error of the Magnetization vs Block Size')
    plt.xlabel('Block Size')
    plt.ylabel('$\sigma$')
    plt.xlim(0,len(xRange))
    fig.tight_layout()
    plt.legend(loc='best')
    plt.show()
    '''
    
#    for i in range(50):
#        plt.plot(xRange,Sigmas[1][i],label='T = ' + str((1+i)/10))
#    plt.legend(loc='best',prop={'size':8})
    
#    fig = plt.figure(1)
#    plt.errorbar(T,M,yerr=np.sqrt(np.var(M)/sweeps),fmt='b-*',label='Data')
#    plt.title('Magnetization vs Temperature')
#    plt.xlabel('Temperature')
#    plt.ylabel('Magnetization')
#    fig.tight_layout()
#    plt.show()
    

    # Plotting a lattice + clusters
    sweep = 3 #TODO loop

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, parameters['lattice_size'][0]])
    ax.set_ylim([0, parameters['lattice_size'][1]])
    assert len(parameters['lattice_size']) == 2
    it = np.nditer(lat_list[sweep], flags=['multi_index'])
    while not it.finished:
        s = it.multi_index

        cluster = lat_bond[sweep][s]
        max_cluster = np.max(lat_bond[sweep])
        color = plt.cm.rainbow(np.linspace(0,1,max_cluster + 1))
        if cluster != 0:
            p2 = patches.Rectangle(s, 1, 1, linewidth=0, fill=True, color=color[cluster], alpha=0.9)
            ax.add_patch(p2)

        if lat_list[sweep][s] == -1:
            p1 = patches.Rectangle(s, 1, 1, linewidth=0, fill=None, hatch='--')
        elif lat_list[sweep][s] == 1:
            p1 = patches.Rectangle(s, 1, 1, linewidth=0, fill=None, hatch='||')
        else:
            p1 = patches.Rectangle(s, 1, 1, linewidth=0, fill=True, color='r', alpha=1)
        ax.add_patch(p1)

        it.iternext()

    plt.show()

    raise NotImplementedError

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


def full_sim(dirname, dirnames):
    '''
    loads all simulations foud in dirnames and runs load_sim() on them
    '''
    #mpl.rcParams.update({'font.size': 22})

    list_all = []

    with tqdm.tqdm(desc= "crunching", total=len(dirnames), dynamic_ncols=True) as bar:
        for subdir in dirnames:
            list_all.append(load_sim(subdir, Plot=False))
            bar.update()

    for algo in ['Monte Carlo', 'Cluster']:
        mask_algo = [i for i,j in enumerate(list_all) if j[0]['mc_algorithm'] == algo]
        mask_temp = [(j[0]['mc_temp'], i) for i,j in enumerate(list_all) if i in mask_algo]
        mask_temp = sorted(mask_temp)
        for temp, sims in itertools.groupby(mask_temp, lambda x: x[0]):
            mask_sim = [k[1] for k in sims]
            #mask_sim := [0, 1, 2, 3, 4, 5, 6, 7, 8, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

            fig, axarr = plt.subplots(6,2,figsize=(20,14))
            plt.suptitle('$k_b T={:1.2f}$'.format(temp))

            for i, init in enumerate(['cold', 'random', 'hot']):
                mask_init = [i for i in mask_sim if list_all[i][0]['lattice_init'] == init]
                #mask_init := [0, 1, 2, 36, 37, 38, 39]

                color = iter(plt.cm.rainbow(np.linspace(0,1,len(mask_init))))
                for j in mask_init:
                    c = next(color)
                    Energy = list_all[j][3]
                    Magnet = list_all[j][4]
                    axarr[2*i, 0].plot(np.arange(len(Energy)), Energy, color=c)
                    axarr[2*i, 0].set_title('Energy vs Time, starting {}'.format(init))
                    axarr[2*i, 0].set_ylim(-2, 0)
                    axarr[2*i+1, 0].plot(np.arange(len(Magnet)), Magnet, color=c)
                    axarr[2*i+1, 0].set_title('Magnetisation vs Time, starting {}'.format(init))
                    axarr[2*i+1, 0].set_ylim(-1, 1)

                    Energy = list_all[j][1]
                    Magnet = list_all[j][2]
                    axarr[2*i, 1].plot(np.arange(len(Energy)), Energy, color=c)
                    axarr[2*i, 1].set_title('Error of Energy vs Time, starting {}'.format(init))
                    axarr[2*i+1, 1].plot(np.arange(len(Magnet)), Magnet, color=c)
                    axarr[2*i+1, 1].set_title('Error of Magnetisation vs Time, starting {}'.format(init))

            fig.tight_layout()
            plt.savefig(os.path.join(dirname, '{}_{}.png'.format(algo, temp)))
            plt.close()

        fig, axarr = plt.subplots(3,2,figsize=(20,14))
        plt.suptitle('{}'.format(algo))

        for temp, sims in itertools.groupby(mask_temp, lambda x: x[0]):
            mask_sim = [k[1] for k in sims]
            #mask_sim := [0, 1, 2, 3, 4, 5, 6, 7, 8, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]


            for i, init in enumerate(['cold', 'random', 'hot']):
                mask_init = [i for i in mask_sim if list_all[i][0]['lattice_init'] == init]
                #mask_init := [0, 1, 2, 36, 37, 38, 39]

                color = iter(plt.cm.rainbow(np.linspace(0,1,len(mask_init))))
                for j in mask_init:
                    c = next(color)

                    err_block = 125
                    block_start = list_all[j][0]['relax_sweeps']

                    Energy = np.sum(list_all[j][3][block_start:])
                    dEnergy = list_all[j][1][err_block]

                    Magnet = np.sum(list_all[j][4][block_start:])
                    dMagnet = list_all[j][2][err_block]

                    axarr[i, 0].errorbar(temp, Energy, yerr=dEnergy, color=c, fmt='o')
                    axarr[i, 0].set_title('Energy vs Temperature, starting {}'.format(init))
                    #axarr[i, 0].set_ylim(-2, 0)
                    axarr[i, 1].errorbar(temp, Magnet, yerr=dMagnet, color=c, fmt='o')
                    axarr[i, 1].set_title('Magnetisation vs Temperature, starting {}'.format(init))
                    #axarr[i, 1].set_ylim(-1, 1)

        fig.tight_layout()
        plt.savefig(os.path.join(dirname, '{}_Temperatures.png'.format(algo)))
        plt.close()

###############################################################################
#           Execute some stuff if directly called                             #
###############################################################################

if __name__ == "__main__":

    if (len(sys.argv) == 2) and (os.path.isfile(sys.argv[1])):
        dirname = os.path.dirname(sys.argv[1])
        run_sim(dirname)
    else:
        print("Stupid you, didn't gave me a config.ini")


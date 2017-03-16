#!/usr/bin/python3

from ising import *
import matplotlib.animation as animation

if __name__ == "__main__":

    parameters = {}

    dirnames = ['./']

    for dirname in dirnames:
        load_config(dirname, parameters)

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
            parameters['mc_temp'] *= 0.5
            T *= 0.5


    savecount = 0
    # Plotting a lattice + clusters
    fig = plt.figure()
    ax = fig.add_subplot(111)
    assert len(parameters['lattice_size']) == 2

    ims = []
    for sweep in range(len(lat_list)):
        ax.set_title('{} algorithm at $k_b T = {}$'.format(parameters['mc_algorithm'], T[0]))
        ax.set_xlim([0, parameters['lattice_size'][0]])
        ax.set_ylim([0, parameters['lattice_size'][1]])
        it = np.nditer(lat_list[sweep], flags=['multi_index'])
        while not it.finished:
            s = it.multi_index

            if parameters['mc_algorithm'] == 'Cluster':
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
            else:
                if lat_list[sweep][s] == -1:
                    p1 = patches.Rectangle(s, 1, 1, linewidth=0, fill=True, color='k')
                elif lat_list[sweep][s] == 1:
                    p1 = patches.Rectangle(s, 1, 1, linewidth=0, fill=True, color='w')
                else:
                    p1 = patches.Rectangle(s, 1, 1, linewidth=0, fill=True, color='r', alpha=1)
                ax.add_patch(p1)

                it.iternext()

        for _ in range(5):
            plt.savefig('videos/pic_{:04d}.png'.format(savecount))
            savecount += 1
        plt.cla()

    #plt.show()

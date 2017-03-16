#!/usr/bin/python3

from ising import *

###############################################################################
#           Execute some stuff if directly called                             #
###############################################################################

if __name__ == "__main__":

    if (len(sys.argv) == 2) and (os.path.isfile(sys.argv[1])):
        dirname = os.path.dirname(sys.argv[1])
        load_sim(dirname, Plot=True)
    elif (len(sys.argv) == 2) and (os.path.isdir(sys.argv[1])) \
                and os.path.split(sys.argv[1])[1] == 'runs':

        dirname = os.path.dirname(sys.argv[1])
        dirnames = []
        for root, dir, file in os.walk(dirname):
            if all(x in file for x in ['config.ini', 'simulation.npz']):
                dirnames.append(root)

        full_sim(dirname, dirnames)

    else:
        print("Stupid you, didn't gave me a config.ini")
        print("If you tried to run on directory leave the tailing /")

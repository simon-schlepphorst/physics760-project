#!/usr/bin/python3

from ising import *

###############################################################################
#           Execute some stuff if directly called                             #
###############################################################################

if __name__ == "__main__":

    if (len(sys.argv) == 2) and (os.path.isfile(sys.argv[1])):
        dirname = os.path.dirname(sys.argv[1])
        load_sim(dirname)
    else:
        print("Stupid you, didn't gave me a config.ini")

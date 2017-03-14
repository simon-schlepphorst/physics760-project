#!/usr/bin/python3

import os
import configparser

def write_config(config, max_sims=10):
    #max_sims shouldn't be higher than 100
    assert max_sims <= 100

    algorithm = config.get('markov chain', 'algorithm').replace(' ', '')
    temperature = 'T_' + config.get('markov chain', 'temperature').replace('.','-')
    init = config.get('lattice', 'init')
    dirname = os.path.join("runs", algorithm, temperature, init)
    #find non-existing folder an create it
    i = 0
    while i < max_sims:
        try:
            os.makedirs(os.path.join(dirname, '{:03d}'.format(i)))
            break
        except FileExistsError:
            i += 1

    if i >= max_sims:
        return False

    #write to file
    filepath = os.path.join(dirname, '{:03d}'.format(i), 'config.ini')
    assert os.path.isfile(filepath) == False

    with open(filepath, 'w') as f:
        config.write(f)

    return True

#open default:
config = configparser.ConfigParser()
with open('config.ini') as f:
    config.read_file(f)

T = [1.0, 1.3, 1.6, 1.8, 2.0, 2.2, 2.3]
#T = [2.4, 2.6, 2.8, 3.0, 3.3, 3.6, 4.0]

for temp in T:
    config.set('markov chain', 'temperature', str(temp))
    for i in ('hot', 'cold', 'random'):
        config.set('lattice', 'init', i)
        while write_config(config, 4):
            pass


'''Creates Principle Layers out of gen-file. Reads input file partition.inp

Inputfile structure:

filename.gen
2    # Number of contacts
iCont1   # Starting index of contact1 atoms (Contact blocks have to be at end)
iCont2
...
rSpecies1 # interaction radius for element 1 in Angstrom (order like gen-file)
rSpecies2
...
'''

import numpy as np
from dptools.gen import Gen


def read_input():
    '''Reads input from partition.inp file-'''
    inputfile = open('partition.inp', 'r')
    lines = inputfile.readlines()
    filename = lines[0][:-1]
    n_contacts = int(lines[1])
    i_contacts = [int(lines[3 + ii]) for ii in range(n_contacts)]
    r_interactions = [float(lines[3 + n_contacts + ii])
                      for ii in range(n_contacts)]
    inputfile.close()
    return filename, i_contacts, r_interactions


def partition():
    '''Creates principle layers-'''
    filename, i_contacts, r_interactions = read_input()
    gen = Gen.fromfile(filename)
    geo = gen.geometry
    natoms = geo.natom
    coords = geo.coords
    species_inds = geo.indexes
    # Creating chains. Zeroth elements are contact bins. Next are device bins.
    # Atom indexes starting with 0
    nchains = len(i_contacts)
    chains = []
    # Adding contacts (generation 0)
    for contact in range(nchains):
        chains.append([])
        if i_contacts[contact] is not max(i_contacts):
            chains[contact].append(np.arange(i_contacts[contact] - 1,
                                             i_contacts[contact + 1] - 1))
        else:
            chains[contact].append(np.arange(i_contacts[contact] - 1, natoms))
    # Each generation creates one more bin in every chain.
    generation = 0
    all_merged = False
    while not all_merged:
        generation += 1
        print('Starting to create generation: ', generation)
        for chain in range(nchains):
            chains[chain].append(create_bin(geo, chain))
        if check_collision(0, 1):
            print('Chains collided. Starting merge.')
            merge_two_chains(chains[0], chains[1])
            all_merged = True


def create_bin(geo, chain):
    '''Creates the next bin in a chain'''
    new_chain = np.array([])
    return new_chain


def check_collision(chainnumber1, chainnumber2):
    '''Checks if two chains collide (having same atoms)'''
    return True


def create_final_chain(chains):
    '''Creates final one dimensional chain out of last two chains'''
    final_chain = np.array([])
    return final_chain


def merge_two_chains(chain1, chain2):
    '''Merges two chains.'''
    merged_chain = np.array([])
    return merged_chain


def save_partition():
    '''Saves created partition'''


if __name__ == '__main__':
    read_input()
    partition()
    save_partition()

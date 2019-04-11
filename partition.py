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
    print('Reading partition.inp')
    inputfile = open('partition.inp', 'r')
    lines = inputfile.readlines()
    filename = lines[0][:-1]
    n_contacts = int(lines[1])
    i_contacts = [int(lines[3 + ii]) for ii in range(n_contacts)]
    r_interactions = [float(lines[3 + n_contacts + ii])
                      for ii in range(n_contacts)]
    inputfile.close()
    return (filename, i_contacts, r_interactions)


def partition(filename, i_contacts, r_interactions):
    '''Creates principle layers'''
    geo = Gen.fromfile(filename).geometry
    interaction_mtrx = _create_interaction_mtrx(r_interactions)
    # Creating chains. Zeroth elements are contact bins. Next are device bins.
    # Atom indexes starting with 1. Careful with getting coordinates and inds!
    nchains = len(i_contacts)
    chains = []
    # Adding contacts (bin_index 0)
    for contact in range(nchains):
        chains.append([])
        if i_contacts[contact] is not max(i_contacts):
            chains[contact].append(set(range(i_contacts[contact],
                                             i_contacts[contact + 1])))
        else:
            chains[contact].append(set(range(i_contacts[contact],
                                             geo.natom + 1)))
    # Each iteration creates one more bin in every chain.
    # Unsorted atoms start with device atoms.
    bin_index = 0
    unsorted = set(range(1, min(i_contacts)))
    all_sorted = False
    while not all_sorted:
        bin_index += 1
        print('Starting to create bins with index: ', bin_index)
        for chain in range(nchains):
            new_bin = _create_bin(chains[chain][bin_index - 1],
                                  unsorted, geo.coords, geo.indexes,
                                  interaction_mtrx)
            chains[chain].append(new_bin)
            unsorted = unsorted - new_bin
        if _check_collision(chains[0], chains[1]):
            print('Chains collided. Starting to merge endpieces')
            if len(chains) == 2:
                chains = _merge_endpiece_bins(chains[0], chains[1])
            all_sorted = True
        # If layers perfectly split, no unsorted atoms are left
        if unsorted == set():
            print('All atoms sorted')
            all_sorted = True
    principle_layers = _create_principle_layers(chains)
    return principle_layers


def _create_interaction_mtrx(r_interactions):
    '''Creates interaction matrix. Gives maximum interaction radius for two
    elements'''
    interaction_mtrx = []
    nspecies = len(r_interactions)
    for rowindex in range(nspecies):
        row = []
        for columnindex in range(nspecies):
            row.append(max([r_interactions[rowindex],
                            r_interactions[columnindex]]))
        interaction_mtrx.append(row)
    return interaction_mtrx

def _create_bin(last_bin, unsorted, coords, species_inds, interaction_mtrx):
    '''Creates the next bin in a chain.'''
    new_bin = set()
    for lastbinatom in last_bin:
        for unsortedatom in unsorted:
            distance = np.linalg.norm(coords[lastbinatom - 1] -
                                      coords[unsortedatom - 1])
            species1 = species_inds[lastbinatom - 1]
            species2 = species_inds[unsortedatom - 1]
            if distance <= interaction_mtrx[species1][species2]:
                new_bin.add(unsortedatom)
    return new_bin


def _check_collision(chain1, chain2):
    '''Checks if two chains collide (having same atoms)'''
    # Merge sets in both chains.
    chain1atoms, chain2atoms = set(), set()
    n_bins = max(len(chain1), len(chain1))
    for binindex in range(n_bins):
        chain1atoms.update(chain1[binindex])
        chain2atoms.update(chain2[binindex])
    # Check if one atom is in both chains.
    if chain1atoms & chain2atoms:
        return True
    return False


def _merge_endpiece_bins(chain1, chain2):
    '''Merges endpiece bins after collision of two chains'''
    chain1[-1] = chain1[-1].union(chain2[-1])
    chain2 = chain2[:-1]
    return [chain1, chain2]


def _merge_two_chains(chain1, chain2):
    '''Merges two chains. Need same length'''
    merged_chain = []
    for binindex in enumerate(chain1):
        merged_chain.append(chain1[binindex[0]].union(chain2[binindex[0]]))
    return merged_chain


def _create_principle_layers(chains):
    '''Creates final one dimensional chain (principle layers) out of last two
    chains. Cuts contacts.'''
    principle_layers = chains[0][1:]
    for binindex in range(len(chains[0]) - 1, 0, -1):
        principle_layers.append(chains[1][binindex])
    print('Principle layers created')
    return principle_layers


def save_partition(principle_layers):
    '''Saves created partition (principle layers). First two lines saves number
    of layers and atoms per layer. From line 3, every line contains the atom
    indexes of one layer.
        nPLs
        nAtomsPL1 nAtomsPL2 ... nAtomsPLn
        PL1_atomindex_1 ... PL1_atomindex_n
        ...
        PLn_atomindex_1 ... PLn_atomindex_n
    '''
    # Create and save line one and two
    outputfile = open('partition.out', 'w')
    outputfile.write(str(len(principle_layers)) + '\n')
    layer_lengths = [str(len(layer)) for layer in principle_layers]
    outputfile.write(' '.join(layer_lengths) + '\n')
    # Sort layers
    for layer in enumerate(principle_layers):
        principle_layers[layer[0]] = list(layer[1])
        principle_layers[layer[0]].sort()
    principle_layers = sorted(principle_layers)
    # Save layers
    for layer in enumerate(principle_layers):
        principle_layers[layer[0]] = list(map(str, layer[1]))
    for layer in enumerate(principle_layers):
        outputfile.write(' '.join(layer[1]) + '\n')
    print('Principle Layers saved to partition.out')
    outputfile.close()


if __name__ == '__main__':
    INPUT = read_input()
    PRINCIPLE_LAYERS = partition(INPUT[0], INPUT[1], INPUT[2])
    save_partition(PRINCIPLE_LAYERS)

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

import sys
import numpy as np
from dptools.gen import Gen


def read_input():
    '''Reads input from partition.inp file-'''
    print('Reading partition.inp')
    inputfile = open('partition.inp', 'r')
    lines = inputfile.readlines()
    filename = lines[0][:-1]
    n_contacts = int(lines[1])
    i_contacts = [int(lines[2 + ii]) for ii in range(n_contacts)]
    i_contacts = sorted(i_contacts)
    r_interactions = [float(lines[2 + n_contacts + ii])
                      for ii in range(len(lines) - n_contacts - 2)]
    inputfile.close()
    return (filename, i_contacts, r_interactions)


def partition(filename, i_contacts, r_interactions, maxiterations):
    '''Creates principle layers'''
    geo = Gen.fromfile(filename).geometry
    interaction_mtrx = _create_interaction_mtrx(r_interactions)
    # Creating chains. Zeroth elements are contact bins. Next are device bins.
    # Atom indexes starting with 1. Careful with getting coordinates and inds!
    nchains = len(i_contacts)
    allchains = []
    # Adding contacts (bin_index 0)
    for contact in range(nchains):
        allchains.append([])
        if i_contacts[contact] is not max(i_contacts):
            allchains[contact].append(set(range(i_contacts[contact],
                                                i_contacts[contact + 1])))
        else:
            allchains[contact].append(set(range(i_contacts[contact],
                                                geo.natom + 1)))
    # Each iteration creates one more bin in every chain.
    # Unsorted atoms start with device atoms.
    bin_index = 0
    unsorted = set(range(1, min(i_contacts)))
    all_sorted = False
    while not all_sorted:
        bin_index += 1
        print('Starting to create bins with index: ', bin_index)
        collision = False
        for chain in range(nchains):
            new_bin = _create_bin(allchains[chain][bin_index - 1],
                                  unsorted, geo.coords, geo.indexes,
                                  interaction_mtrx)
            allchains[chain].append(new_bin)
            if _check_collision(allchains[0], allchains[1]):
                collision = True
            # Only modify unsorted if all bins are created
            # There will be no collision otherwise
            if chain == (nchains - 1):
                for chainagain in allchains:
                    unsorted = unsorted - chainagain[-1]
        # If layers perfectly split, no unsorted atoms are left
        if collision:
            print('Chains collided. Starting to merge endpieces')
            if len(allchains) == 2:
                allchains = _merge_endpiece_bins(allchains[0], allchains[1])
        if unsorted == set():
            print('All atoms sorted')
            all_sorted = True
        # Max binnumber in case of infinite loop
        if bin_index >= maxiterations:
            print('Maximum iterations reached: ' + str(maxiterations))
            print('Interaction radii might be too short')
            print('Exiting program')
            sys.exit()
    principle_layers = _create_principle_layers(allchains)
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
    for currentbin in enumerate(chain1):
        chain1atoms.update(chain1[currentbin[0]])
    for currentbin in enumerate(chain2):
        chain2atoms.update(chain2[currentbin[0]])
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
    for binindex in range(len(chains[1]) - 1, 0, -1):
        principle_layers.append(chains[1][binindex])
    print('Principle layers created')
    return principle_layers


def test_interaction(filename, i_contacts, r_interactions, principle_layers):
    '''Tests if every principle layer only interacts with adjacent layers.
    Also checks that contacts only interact with one layer. Does not check if
    contacts are interacting.'''
    geo = Gen.fromfile(filename).geometry
    interaction_mtrx = _create_interaction_mtrx(r_interactions)
    test_passed = True
    # Check layers
    for layer in range(1, len(principle_layers) - 1):
        interacting_layers = set()
        for currentlayeratom in principle_layers[layer]:
            for otherlayer in enumerate(principle_layers):
                for otheratom in principle_layers[otherlayer[0]]:
                    distance = np.linalg.norm(geo.coords[otheratom - 1] -
                                              geo.coords[currentlayeratom - 1])
                    species1 = geo.indexes[currentlayeratom - 1]
                    species2 = geo.indexes[otheratom - 1]
                    if distance <= interaction_mtrx[species1][species2]:
                        interacting_layers.add(otherlayer[0])
            if (min(interacting_layers) < layer - 1 or
                    max(interacting_layers) > layer + 1):
                print('Principle layers were not sorted right! Atom '
                      + str(currentlayeratom) + ' in layer '
                      + str(layer) +
                      ' is interacting with non adjacent layer. '
                      'Interacting with layers: ' + str(interacting_layers))
                test_passed = False
    # Checks if every contact only interacts with one layer
    i_contacts = sorted(i_contacts)
    i_contacts.append(geo.natom + 1)
    for contact in range(len(i_contacts) - 1):
        contactatoms = set(range(i_contacts[contact], i_contacts[contact + 1]))
        interacting_layers = set()
        for atom in contactatoms:
            for layer in enumerate(principle_layers):
                for layeratom in principle_layers[layer[0]]:
                    distance = np.linalg.norm(geo.coords[atom - 1] -
                                              geo.coords[layeratom - 1])
                    species1 = geo.indexes[atom - 1]
                    species2 = geo.indexes[layeratom - 1]
                    if distance <= interaction_mtrx[species1][species2]:
                        interacting_layers.add(layer[0])
            if len(interacting_layers) > 1:
                print('Principle layers were not sorted right! Atom '
                      + str(atom) + ' in contact '
                      + str(contact) + ' is interacting with more than one '
                      + 'layers: ' + str(interacting_layers))
                test_passed = False
    if test_passed:
        print('Interaction test passed')
    return test_passed


def create_jmolscript(filename, principle_layers):
    '''Creates jmol script for alternated coloring of created principle layers.
    Also labels layers. Script can be opened in jmol. Jmol also needs and .xyz
    format of the geometry (use gen2xyz fileame.gen).
    Run with: jmol -s scriptname'''
    scriptname = filename[:-4] + '.part.js'
    jmolscript = open(scriptname, 'w')
    jmolscript.write('load ' + filename[:-4] + '.xyz' + '\n')
    for layer in enumerate(principle_layers):
        for atom in layer[1]:
            jmolscript.write('select atomno=' + str(atom) + '\n')
            jmolscript.write('label ' + str(layer[0]) + '\n')
            if layer[0]%2 == 0:
                color = 'yellow'
            else:
                color = 'red'
            jmolscript.write('color ' + color + '\n')
    jmolscript.close()
    print('Jmol script for graphical view created: ' + scriptname)


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
    MAXITERATIONS = 200
    FILENAME, I_CONTACTS, R_INTERACTION = read_input()
    PRINCIPLE_LAYERS = partition(FILENAME, I_CONTACTS, R_INTERACTION,
                                 MAXITERATIONS)
    if test_interaction(FILENAME, I_CONTACTS, R_INTERACTION, PRINCIPLE_LAYERS):
        save_partition(PRINCIPLE_LAYERS)
    create_jmolscript(FILENAME, PRINCIPLE_LAYERS)

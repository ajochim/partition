'''Creates Principle Layers out of gen-file and further information in
partition.inp. Principle layers are used for transport calculation input in
the dftb+ software. A principle layer is a contigous group of atoms only
interacting with adjacent layers. Every principle layer has exactly one
following neighbour, therefore they are a kind of onedimensional structure.

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
    '''Creates principle layers. Atom indexes starting with 1 (Geo object
    coordinates and inds with 0!) Algorithm first creates chains starting from
    every contact. Each chain element is called a bin and every bin only
    interacts with adjacent bins or contacts (Not a principle layer yet).
    Chains (and also dead ends) are merged until only two chains are left.
    Those two chains will be resorted to create the principle layers.'''
    geo = Gen.fromfile(filename).geometry
    allchains = _create_starting_chains(i_contacts, geo.natom + 1)
    unsorted_atoms = set(range(1, min(i_contacts)))
    bin_index = 0
    all_sorted = False
    interaction_mtrx = _create_interaction_mtrx(r_interactions)
    print('Starting to create chains from every contact:')
    while not all_sorted:
        if bin_index >= maxiterations:
            print('Maximum iterations reached:', maxiterations)
            print('Exiting program')
            sys.exit()
        bin_index += 1
        allchains, unsorted_atoms = _create_next_bins(allchains,
                                                      unsorted_atoms,
                                                      geo,
                                                      interaction_mtrx,
                                                      bin_index)
        merged = False
        if len(allchains) > 2:
            allchains, merged = (
                _merge_untill_nointeracting(allchains, geo, interaction_mtrx))
        print('Bins for iteration', bin_index, 'created. Chains merged:',
              merged)
        if _bins_interact(allchains[0][-1], allchains[1][-1], geo.coords,
                          geo.indexes, interaction_mtrx):
            print('Starting final steps:')
            if _check_collision(allchains[0], allchains[1]):
                allchains[0][-1] = allchains[0][-1].union(allchains[1][-1])
                del allchains[1][-1]
                print('Merged last bins due to collision')
            if unsorted_atoms != set():
                allchains, unsorted_atoms = _sort_remaining(allchains,
                                                            unsorted_atoms,
                                                            geo,
                                                            interaction_mtrx)
                print('Sorted remaining atoms')
        if unsorted_atoms == set():
            print('All atoms sorted')
            all_sorted = True
    principle_layers = _create_finalchain(allchains)
    print('Principle layers created (Last two chains combined)')
    return principle_layers


def _create_interaction_mtrx(r_interactions):
    '''Creates interaction matrix. Gives maximum interaction radius for two
    elements'''
    interaction_mtrx = []
    nspecies = len(r_interactions)
    for row_index in range(nspecies):
        row = []
        for column_index in range(nspecies):
            row.append(max([r_interactions[row_index],
                            r_interactions[column_index]]))
        interaction_mtrx.append(row)
    return interaction_mtrx


def _create_starting_chains(i_contacts, natoms):
    '''Adding contact atoms as starting elements (bin_index=0) to chains.'''
    allchains = []
    nchains = len(i_contacts)
    for contact in range(nchains):
        allchains.append([])
        if i_contacts[contact] is not max(i_contacts):
            allchains[contact].append(set(range(i_contacts[contact],
                                                i_contacts[contact + 1])))
        else:
            allchains[contact].append(set(range(i_contacts[contact],
                                                natoms)))
    return allchains


def _create_next_bins(allchains, unsorted_atoms, geo, interaction_mtrx,
                      bin_index):
    '''Creates the next bins in a chain. Also updates unsorted. If new bin is
    empty, atom with smallest distance creates new bin (distance_dict
    created to accomplish this)'''
    for chain in enumerate(allchains):
        new_bin = set()
        distance_dict = {}
        for lastbinatom in allchains[chain[0]][bin_index - 1]:
            for unsortedat in unsorted_atoms:
                distance = np.linalg.norm(geo.coords[lastbinatom - 1] -
                                          geo.coords[unsortedat - 1])
                distance_dict[lastbinatom, unsortedat] = distance
                species = (geo.indexes[lastbinatom - 1],
                           geo.indexes[unsortedat - 1])
                if distance <= interaction_mtrx[species[0]][species[1]]:
                    new_bin.add(unsortedat)
        if new_bin == set():
            new_bin = set([min(distance_dict, key=distance_dict.get)[1]])
        allchains[chain[0]].append(new_bin)
    # Remove atoms from unsorted_atoms
    for chain in allchains:
        for current_bin in chain:
            unsorted_atoms = unsorted_atoms - current_bin
    return allchains, unsorted_atoms


def _merge_two_chains(chain1, chain2):
    '''Merges two chains. Needs same length'''
    merged_chain = []
    for bin_index in enumerate(chain1):
        merged_chain.append(chain1[bin_index[0]].union(chain2[bin_index[0]]))
    return merged_chain


def _bins_interact(bin1, bin2, coords, species_inds, interaction_mtrx):
    '''Checks if two bins are interacting'''
    interaction = False
    for bin1atom in bin1:
        for bin2atom in bin2:
            distance = np.linalg.norm(coords[bin1atom - 1] -
                                      coords[bin2atom - 1])
            species1 = species_inds[bin1atom - 1]
            species2 = species_inds[bin2atom - 1]
            if distance <= interaction_mtrx[species1][species2]:
                interaction = True
    return interaction


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


def _merge_untill_nointeracting(allchains, geo, interaction_mtrx):
    '''Merges all chains untill there is no interacting of last bins or only
    two chains are eft.'''
    interacting = True
    while interacting and len(allchains) > 2:
        interacting = False
        nchains = len(allchains)
        for chain_number_i in range(nchains - 1):
            for chain_number_j in range(chain_number_i + 1, nchains):
                if (not interacting and
                        _bins_interact(allchains[chain_number_i][-1],
                                       allchains[chain_number_j][-1],
                                       geo.coords, geo.indexes,
                                       interaction_mtrx)):
                    merged_chain = _merge_two_chains(allchains[chain_number_i],
                                                     allchains[chain_number_j])
                    allchains[chain_number_i] = merged_chain
                    del allchains[chain_number_j]
                    interacting = True
    return allchains, interacting


def _sort_remaining(allchains, unsorted_atoms, geo, interaction_mtrx):
    '''Merges rest. Creates new chain out of remaining atoms. Merges its bins
    untill the chain is small enough to merge it to a chain from a contact.
    Uses nested strucutre in newchain to use _create_next_bins function.
    '''
    newchain = [[allchains[0][-1]]]
    bin_index = 0
    while unsorted_atoms != set():
        bin_index += 1
        newchain, unsorted_atoms = _create_next_bins(newchain,
                                                     unsorted_atoms, geo,
                                                     interaction_mtrx,
                                                     bin_index)
    while len(newchain[0]) > len(allchains[0]) - 1:
        for bin_index in range(0, len(newchain[0])//2):
            newchain[0][bin_index] = (
                newchain[0][bin_index].union(newchain[0][bin_index + 1]))
            del newchain[0][bin_index + 1]
    allchains[0][-1] = allchains[0][-1].union(newchain[0][0])
    for bin_index in enumerate(newchain[0][1:]):
        reverse_index = -1*bin_index[0] - 1
        allchains[1][reverse_index] = (
            allchains[1][reverse_index].union(newchain[0][bin_index[0] + 1]))
    return allchains, unsorted_atoms


def _create_finalchain(allchains):
    '''Creates final one dimensional chain (principle layers) out of last two
    chains. Cuts contacts.'''
    finalchain = allchains[0][1:]
    for bin_index in range(len(allchains[1]) - 1, 0, -1):
        finalchain.append(allchains[1][bin_index])
    return finalchain


def test_interaction(filename, i_contacts, r_interactions, principle_layers):
    '''Tests if every principle layer only interacts with adjacent layers.
    Also checks that contacts only interact with one layer. Does not check if
    contacts are interacting.'''
    geo = Gen.fromfile(filename).geometry
    interaction_mtrx = _create_interaction_mtrx(r_interactions)
    test_passed = True
    # Check layers
    print('Starting interaction test:')
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
                      ' might be interacting with non adjacent layer. '
                      'Interacting possible with layers: '
                      + str(interacting_layers))
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
        print('Principle layers passed interaction test')
    return test_passed


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
    # Save layers
    for layer in enumerate(principle_layers):
        principle_layers[layer[0]] = list(map(str, layer[1]))
    for layer in enumerate(principle_layers):
        outputfile.write(' '.join(layer[1]) + '\n')
    print('Principle layers saved to partition.out')
    outputfile.close()


def create_jmolscript(filename, principle_layers, labelatoms=False):
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
            atomlabel = ''
            if labelatoms:
                atomlabel = '#' + str(atom)
            jmolscript.write('label ' + str(layer[0] + 1) + atomlabel + '\n')
            if layer[0]%2 == 0:
                color = 'yellow'
            else:
                color = 'red'
            jmolscript.write('color ' + color + '\n')
    jmolscript.close()
    print('Jmol script for graphical view created: ' + scriptname)

if __name__ == '__main__':
    MAXITERATIONS = 100
    print('')
    FILENAME, I_CONTACTS, R_INTERACTION = read_input()
    print('')
    PRINCIPLE_LAYERS = partition(FILENAME, I_CONTACTS, R_INTERACTION,
                                 MAXITERATIONS)
    print('')
    if test_interaction(FILENAME, I_CONTACTS, R_INTERACTION, PRINCIPLE_LAYERS):
        print('')
        save_partition(PRINCIPLE_LAYERS)
    print('')
    create_jmolscript(FILENAME, PRINCIPLE_LAYERS, labelatoms=False)

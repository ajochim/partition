'''Creates Principle Layers out of gen-file. Reads input file partition.inp

Inputfile structure:

filename.gen
2    # Number of contacts
iCont1   # Starting index of contact1 atoms (Contact block has to be at end)
iCont2
...
rSpecies1 # interaction radius for element 1 in Angstrom
rSpecies2
...
'''

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
#    natoms = geo.natom
#    coords = geo.coords
#    inds = geo.indexes

def save_partition():
    '''Saves created partition'''
    pass


if __name__ == '__main__':
    read_input()
    partition()
    save_partition()

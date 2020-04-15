import os
import os.path

import utils
import numpy as np
import json
import sys

dest = '././output'
if not os.path.isdir(dest):
    os.makedirs('././output')

args = sys.argv[1:]

if len(args) != 0:
    print('Using Ikoria Data')
    cubes = np.load('././ml_files/cubes.npy')
    fname = '././output/iko_adj_mtx.npy'
else:
    map_file = '././data/maps/nameToId.json'
    folder = "././data/cube/"
    print('getting data')
    num_cards, name_lookup, card_to_int, int_to_card = \
        utils.get_card_maps(map_file)

    num_cubes = utils.get_num_cubes(folder)

    cubes = utils.build_cubes(folder, num_cubes, num_cards, name_lookup,
                            card_to_int)
    fname = '././output/full_adj_mtx.npy'

    with open('././output/int_to_card.json', 'w') as out_lookup:
        json.dump(int_to_card, out_lookup)

print('creating matrix')
adj_mtx = utils.create_adjacency_matrix(cubes)

with open(fname, 'wb') as out_mtx:
    np.save(out_mtx, adj_mtx)

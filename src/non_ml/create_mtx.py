import os
import os.path

import utils
import numpy as np
import json

map_file = '././data/maps/nameToId.json'
folder = "././data/cube/"
print('getting data')
num_cards, name_lookup, card_to_int, int_to_card = \
    utils.get_card_maps(map_file)

num_cubes = utils.get_num_cubes(folder)

cubes = utils.build_cubes(folder, num_cubes, num_cards, name_lookup,
                          card_to_int)
print('creating matrix')
adj_mtx = utils.create_adjacency_matrix(cubes)

dest = '././output'
if not os.path.isdir(dest):
    os.makedirs('././output')

with open('././output/full_adj_mtx.npy', 'wb') as out_mtx:
    np.save(out_mtx, adj_mtx)

with open('././output/int_to_card.json', 'w') as out_lookup:
    json.dump(int_to_card, out_lookup)

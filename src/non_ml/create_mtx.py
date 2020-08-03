import utils
import numpy as np
import json
from pathlib import Path

map_file = '././data/maps/nameToId.json'
folder = "././data/deck/"
require_side = False
print('getting data')
num_cards, name_lookup, card_to_int, int_to_card = \
    utils.get_card_maps(map_file)

num_cubes = utils.get_num_cubes(folder, require_side=require_side)

print(f'num cubes: {num_cubes}')

cubes = utils.build_decks(folder, num_cubes, num_cards,
                          name_lookup, card_to_int, require_side=require_side)

print('creating matrix')
adj_mtx = utils.create_adjacency_matrix(cubes)

Path(f'././output').mkdir(parents=True, exist_ok=True)
with open('././output/full_adj_mtx.npy', 'wb') as out_mtx:
    np.save(out_mtx, adj_mtx)

with open('././output/int_to_card.json', 'w') as out_lookup:
    json.dump(int_to_card,  out_lookup)

import utils
import numpy as np
import json

map_file = 'data/maps/nameToId.json'
folder = "data/cube/"

num_cards,name_lookup,card_to_int,int_to_card = utils.get_card_maps(map_file)

num_cubes = utils.get_num_cubes(folder)

cubes = utils.build_cubes(folder, num_cubes, num_cards, name_lookup, card_to_int)

adj_mtx = utils.create_adjacency_matrix(cubes)

out_mtx = open('output/full_adj_mtx.npy','wb')
np.save(out_mtx,adj_mtx)

out_lookup = open('output/int_to_card.json','w')
json.dump(int_to_card,out_lookup)



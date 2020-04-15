import os
import os.path
import urllib.request
import unidecode
import utils
import numpy as np
import json

def get_cards(cube_name, root = "https://cubecobra.com"):
    url = root + "/cube/api/cubelist/" + cube_name
    fp = urllib.request.urlopen(url)
    mybytes = fp.read()
    mystr = mybytes.decode("utf8")
    fp.close()
    return mystr.split("\n")

map_file = '././data/maps/nameToId.json'
folder = "././data/cube/"
print('getting data')
num_cards, name_lookup, card_to_int, int_to_card = \
    utils.get_card_maps(map_file)

num_cubes = utils.get_num_cubes(folder)

card_to_int = dict()
card_max = 0

cubes = np.zeros((num_cubes,num_cards))
counter = 0
for f in os.listdir(folder):
    full_path = os.path.join(folder,f)
    contents = json.load(open(full_path,'rb'))
    for cube in contents:
        identifier = cube['_id']
        print(counter,identifier)
        cards = get_cards(identifier)
        for card in cards:
            card_idx = card_to_int.get(card)
            if card_idx is None:
                card_idx = card_max
                card_to_int[card] = card_max
                card_max += 1
            cubes[counter,card_idx] = 1
        counter+=1

int_to_card = {v:k for k,v in card_to_int.items()}

dest = '././output'
if not os.path.isdir(dest):
    os.makedirs('././output')
with open('././output/cubes.npy', 'wb') as out_mtx:
    np.save(out_mtx, cubes)
with open('././output/int_to_card_new.json', 'w') as out_lookup:
    json.dump(int_to_card, out_lookup)
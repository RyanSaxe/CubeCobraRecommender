import os
import os.path
import urllib.request
import unidecode
import utils
import numpy as np
import json

def get_cards(cube_name, root = "https://cubecobra.com", timeout=0.5):
    url = root + "/cube/api/cubelist/" + cube_name
    fp = urllib.request.urlopen(url,timeout=0.5)
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
#starting with 22k and will downsize to right size
cubes = np.zeros((num_cubes,22000))
old_cubes = dict()
counter = 0
for f in os.listdir(folder):
    full_path = os.path.join(folder,f)
    contents = json.load(open(full_path,'rb'))
    for cube in contents:
        identifier = cube['_id']
        print(counter + 1,"/",num_cubes)
        try:
            cards = get_cards(identifier)
        except:
            print("\t this cube was deleted")
            cards = [name_lookup.get(card['cardID']) for card in cube['cards']]
            old_cubes[counter] = cards
            counter += 1
            continue
        for card in cards:
            card_idx = card_to_int.get(card)
            if card_idx is None:
                card_idx = card_max
                card_to_int[card] = card_max
                card_max += 1
            cubes[counter,card_idx] = 1
        counter+=1

for row,cube in old_cubes.items():
    for card in cube:
        card_idx = card_to_int.get(card)
        if card_idx is not None:
            cubes[row,card_idx] = 1

int_to_card = {v:k.lower() for k,v in card_to_int.items()}

cubes = cubes[:,:card_max]

dest = '././output'
if not os.path.isdir(dest):
    os.makedirs('././output')
with open('././ml_files/cubes.npy', 'wb') as out_mtx:
    np.save(out_mtx, cubes)
with open('././ml_files/iko_id_map.json', 'w') as out_lookup:
    json.dump(int_to_card, out_lookup)
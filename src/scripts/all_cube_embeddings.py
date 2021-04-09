import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import unidecode
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CosineSimilarity

args = sys.argv[1:]
model_dir = Path('ml_files/20210409')

non_json = True
root = "https://cubecobra.com"
with open(model_dir / 'int_to_card.json', 'rb') as map_file:
    int_to_card = json.load(map_file)
card_to_int = {v:k for k,v in enumerate(int_to_card)}
num_cards = len(int_to_card)

base_url = root + "/cube/api/cubelist"

#def get_cube_cards(cubeid):
#    url = f'{base_url}/{cubeid}'
#    with urllib.request.urlopen(url) as request:
#        mybytes = request.read()
#    mystr = mybytes.decode("utf8")

#    card_names = mystr.split("\n")

#    cube_indices = []
#    for name in card_names:
#        idx = card_to_int.get(unidecode.unidecode(name.lower()))
#        #skip unknown cards (e.g. custom cards)
#        if idx is not None:
#            cube_indices.append(idx)
#    cube = np.zeros(num_cards)
#    return cube
cube_folder = Path('data/cubes/')
num_objs = 0
for filename in cube_folder.iterdir():
    with open(filename, 'rb') as obj_file:
        contents = json.load(obj_file)
    num_objs += len([obj for obj in contents if obj['numDecks'] > 0 or len(obj['cards']) >= 90])
cubes = np.zeros((num_objs, num_cards))
cube_ids = ['' for _ in range(num_objs)]
counter = 0
for filename in cube_folder.iterdir():
    with open(filename, 'rb') as cube_file:
        contents = json.load(cube_file)
    for cube in contents:
        if cube['numDecks'] > 0 or len(cube['cards']) >= 90:
            card_ids = []
            for card_name in cube['cards']:
                if card_name is not None:
                    card_id = card_to_int.get(card_name, None)
                    if card_id is not None:
                        card_ids.append(card_id)
            cubes[counter, card_ids] = 1
            cube_ids[counter] = cube['id']
            counter += 1
cards = np.zeros((num_cards,num_cards))
np.fill_diagonal(cards, 1)

model = load_model(model_dir)

cube_embs = model.encoder(cubes).numpy()
card_embs = model.encoder(cards).numpy()

cube_emb_tsv = '\n'.join('\t'.join(str(x) for x in cube_emb) for cube_emb in cube_embs)
card_emb_tsv = '\n'.join('\t'.join(str(x) for x in card_emb) for card_emb in card_embs)
with open(model_dir / 'embedding.tsv', 'w') as embedding_file:
    embedding_file.write(f'{cube_emb_tsv}\n{card_emb_tsv}')

cube_label_tsv = '\n'.join(f'{cube_id}\tCube' for cube_id in cube_ids)
card_label_tsv = '\n'.join(f'{card_name}\tCard' for card_name in int_to_card)
with open(model_dir / 'embedding_labels.tsv', 'w') as embedding_labels_file:
    embedding_labels_file.write(f'Name/Id\tType\n{cube_label_tsv}\n{card_label_tsv}')

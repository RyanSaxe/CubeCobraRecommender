import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import unidecode
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CosineSimilarity

args = sys.argv[1:]
cube_name = args[0].replace('_',' ')
N = int(args[1])
model_dir = Path('ml_files/20210409')

non_json = True
root = "https://cubecobra.com"
with open(model_dir / 'int_to_card.json', 'rb') as map_file:
    int_to_card = json.load(map_file)
int_to_card = {int(k):v for k,v in enumerate(int_to_card)}
card_to_int = {v:k for k,v in int_to_card.items()}

num_cards = len(int_to_card)

url = root + "/cube/api/cubelist/" + cube_name

with urllib.request.urlopen(url) as request:
    mybytes = request.read()
mystr = mybytes.decode("utf8")

card_names = mystr.split("\n")

cube_indices = []
for name in card_names:
    idx = card_to_int.get(unidecode.unidecode(name.lower()))
    #skip unknown cards (e.g. custom cards)
    if idx is not None:
        cube_indices.append(idx)
cube = np.zeros((1, num_cards))
cube[0, cube_indices] = 1

model = load_model(model_dir)

cards = np.zeros((num_cards,num_cards))
np.fill_diagonal(cards,1)

dist_f = CosineSimilarity()

embs = model.encoder(cards)
cube_emb = model.encoder(cube)[0]

dists = np.array([
    dist_f(cube_emb,x).numpy() for x in embs
])

ranked = dists.argsort()

for i in range(N):
    card_idx = ranked[i]
    print(str(i + 1) + ":",int_to_card[card_idx],dists[card_idx])

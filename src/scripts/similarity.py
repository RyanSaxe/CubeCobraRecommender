import json
import sys
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CosineSimilarity

args = sys.argv[1:]
name = args[0].replace('_',' ')
N = int(args[1])
model_dir = Path('ml_files/20210408/model')

with open(model_dir / 'int_to_card.json', 'rb') as map_file:
    int_to_card = json.load(map_file)
int_to_card = {int(k):v for k,v in enumerate(int_to_card)}
card_to_int = {v:k for k,v in int_to_card.items()}

num_cards = len(int_to_card)

model = load_model(model_dir)

cards = np.zeros((num_cards,num_cards))
np.fill_diagonal(cards,1)

dist_f = CosineSimilarity()

embs = model.encoder(cards)
idx = card_to_int[name]

dists = np.array([
    dist_f(embs[idx],x).numpy() for x in embs
])

ranked = dists.argsort()

for i in range(N):
    card_idx = ranked[i]
    print(str(i + 1) + ":",int_to_card[card_idx],dists[card_idx])

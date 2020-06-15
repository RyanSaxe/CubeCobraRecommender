import json
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CosineSimilarity
import sys
import numpy as np

args = sys.argv[1:]
name1 = args[0].replace('_',' ')
name2 = args[1].replace('_',' ')

int_to_card = json.load(open('ml_files/recommender_id_map.json','r'))
int_to_card = {int(k):v for k,v in int_to_card.items()}
card_to_int = {v:k for k,v in int_to_card.items()}

num_cards = len(int_to_card)

model = load_model('ml_files/recommender')

card1 = np.zeros(num_cards)
card1[card_to_int[name1]] = 1

card2 = np.zeros(num_cards)
card2[card_to_int[name2]] = 1

dist_f = CosineSimilarity()

embs = model.encoder([card1,card2])

print(
    dist_f(embs[0],embs[1]).numpy()
)
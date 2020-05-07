import json
import numpy as np
import unidecode
from tensorflow import keras
import urllib.request

def get_ml_embeddings(cards, n_decimals=5):
    int_to_card = json.load(open("./ml_files/recommender_id_map.json", "r"))
    int_to_card = {int(k): v for k, v in int_to_card.items()}
    card_to_int = {v: k for k, v in int_to_card.items()}

    num_cards = len(int_to_card)

    model = keras.models.load_model('ml_files/recommender')

    card_mtx = np.zeros((len(cards),num_cards))

    for i,card in enumerate(cards):
        idx = card_to_int[card]
        card_mtx[i,idx] = 1
    
    embeddings = model.encoder(card_mtx).numpy()
    rounded_embs = np.round(embeddings,n_decimals)
    return rounded_embs.tolist()


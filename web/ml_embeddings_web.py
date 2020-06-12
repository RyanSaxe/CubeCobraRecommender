import json
import numpy as np
import unidecode
from tensorflow import keras

int_to_card = json.load(open("./ml_files/recommender_id_map.json", "r"))
int_to_card = {int(k): v for k, v in int_to_card.items()}
card_to_int = {v: k for k, v in int_to_card.items()}

model = keras.models.load_model('ml_files/recommender')


def get_ml_embeddings(cards, n_decimals=5):

    num_cards = len(int_to_card)

    card_mtx = np.zeros((len(cards), num_cards))

    doesnt_exist = []

    for i, card in enumerate(cards):
        if card is None:
            doesnt_exist.append(i)
            continue
        idx = card_to_int.get(unidecode.unidecode(card.lower()))
        if idx is not None:
            card_mtx[i, idx] = 1
        else:
            doesnt_exist.append(i)

    embeddings = model.encoder(card_mtx).numpy()

    for idx in doesnt_exist:
        embeddings[idx, :] = np.zeros_like(embeddings[idx, :])

    rounded_embs = np.round(embeddings, n_decimals)
    return rounded_embs.tolist()

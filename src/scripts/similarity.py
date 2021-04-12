import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

DEFAULT_COUNT = 25
DEFAULT_MODEL = '20210409'
DEFAULT_JSON = False
DEFAULT_ROOT = 'https://cubecobra.com'

def find_card_similarities(card, count=DEFAULT_COUNT, model=DEFAULT_MODEL, use_json=DEFAULT_JSON, root=DEFAULT_ROOT):
    model_dir = Path('ml_files') / model

    with open(model_dir / 'int_to_card.json', 'rb') as map_file:
        int_to_card = json.load(map_file)
    card_to_int = {v: k for k, v in enumerate(int_to_card)}
    num_cards = len(int_to_card)

    loaded_model = load_model(model_dir)
    cards = np.zeros((num_cards,num_cards))
    np.fill_diagonal(cards,1)
    embs = loaded_model.encoder(cards)
    idx = card_to_int[card.lower()]
    dists = tf.reshape(tf.keras.layers.dot([tf.repeat(tf.expand_dims(embs[idx], 0), len(embs), axis=0), embs], axes=1, normalize=True), -1).numpy()

    ranked = dists.argsort()[::-1]

    high = [(int_to_card[x], dists[x]) for x in ranked[:count]]
    low = [(int_to_card[x], dists[x]) for x in ranked[-count:]]

    if use_json:
        return {
            'high': {name: value for name, value in high},
            'low': {name: value for name, value in low},
        }
    else:
        high_str = '\n'.join(f'{name}: {value}' for name, value in high)
        low_str = '\n'.join(f'{name}: {value}' for name, value in low)
        return f'{high_str}\n...\n{low_str}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--card', '-c', help='The full name of the card to find similarities for.')
    parser.add_argument('--count', '--number', '-n', default=DEFAULT_COUNT, type=int, help='The number of recommended cuts and additions to recommend.')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL, help='The path under ml_files to the model to use for recommendations.')
    parser.add_argument('--json', dest='use_json', action='store_true', help='Output the results as json instead of plain text.')
    parser.add_argument('--root', default=DEFAULT_ROOT, help='The base url for the CubeCobra instance to retrieve the cube from.')
    args = parser.parse_args()

    print(find_card_similarities(**vars(args)))

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import unidecode
from tensorflow.keras.models import load_model

DEFAULT_COUNT = 25
DEFAULT_MODEL = '20210409'
DEFAULT_JSON = False
DEFAULT_ROOT = 'https://cubecobra.com'

def recommend_changes(cube, count=DEFAULT_COUNT, model=DEFAULT_MODEL, use_json=DEFAULT_JSON, root=DEFAULT_ROOT):
    print('Getting Cube List . . . \n')

    url = root + "/cube/api/cubelist/" + cube

    with urllib.request.urlopen(url) as request:
        mybytes = request.read()
    mystr = mybytes.decode("utf8")

    card_names = mystr.split("\n")
    model_dir = Path('ml_files') / model

    print ('Loading Card Name Lookup . . . \n')

    with open(model_dir / 'int_to_card.json', 'rb') as map_file:
        int_to_card = json.load(map_file)
    card_to_int = {v:k for k,v in enumerate(int_to_card)}

    num_cards = len(int_to_card)

    print ('Creating Cube Vector . . . \n')

    cube_indices = [card_to_int[unidecode.unidecode(name.lower())]
                    for name in card_names if unidecode.unidecode(name.lower()) in card_to_int]
    one_hot_cube = np.zeros(num_cards)
    one_hot_cube[cube_indices] = 1

    print(f'Loading Model {model_dir}. . . \n')

    loaded_model = load_model(model_dir)

    print ('Generating Recommendations . . . \n')

    one_hot_cube = one_hot_cube.reshape(1, num_cards)
    results = loaded_model.decoder(loaded_model.encoder(one_hot_cube))[0].numpy()

    ranked = results.argsort()[::-1]

    output = {
        'additions':dict(),
        'cuts':dict(),
    }
    output_str = ''

    cuts = []
    adds = []
    for i, rec in enumerate(ranked):
        card = int_to_card[rec]
        if one_hot_cube[0][rec] == 0 and len(adds) < count:
            adds.append((card, results[rec]))
        elif one_hot_cube[0][rec] == 1:
            cuts.append((card, results[rec]))
    cuts = cuts[-count:]
    if use_json:
        return json.dumps({
            'additions': {name: value.item() for name, value in adds},
            'cuts': {name: value.item() for name, value in cuts}
        })
    else:
        adds_str = '\n'.join(f'{name}: {value}' for name, value in adds)
        cuts_str = '\n'.join(f'{name}: {value}' for name, value in cuts)
        return f'{adds_str}\n...\n{cuts_str}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cube', '-c', help='The id or short name of the cube to recommend for.')
    parser.add_argument('--count', '--number', '-n', default=DEFAULT_COUNT, type=int, help='The number of recommended cuts and additions to recommend.')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL, help='The path under ml_files to the model to use for recommendations.')
    parser.add_argument('--json', dest='use_json', action='store_true', help='Output the results as json instead of plain text.')
    parser.add_argument('--root', default=DEFAULT_ROOT, help='The base url for the CubeCobra instance to retrieve the cube from.')
    args = parser.parse_args()

    print(recommend_changes(**vars(args)))

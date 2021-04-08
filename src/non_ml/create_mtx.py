import json
from pathlib import Path

import numpy as np

import src.non_ml.utils as utils


def is_valid_deck(deck):
    # return True
    return len(deck['side']) > 0


if __name__ == '__main__':
    data = Path('data')
    maps = data / 'maps'
    decks_folder = data / 'decks'
    with open(maps / 'carddict.json', 'rb') as carddict_file:
        carddict = json.load(carddict_file)
    print('Loading Deck Data.')
    num_decks = 0
    seen_cards = set()
    for filename in decks_folder.iterdir():
        with open(filename, 'rb') as fp:
            contents = json.load(fp)
        for deck in contents:
            card_ids = []
            if is_valid_deck(deck):
                num_decks += 1
                for card_name in deck['main']:
                    if card_name is not None:
                        seen_cards.add(card_name)
    print(f'There are {num_decks} decks.')
    int_to_card = list(seen_cards)
    card_to_int = {v: i for i, v in enumerate(int_to_card)}

    decks = utils.build_decks(decks_folder, num_decks, len(int_to_card),
                              card_to_int, validation_func=is_valid_deck,
                              soft_validation=0)

    print('creating matrix')
    adj_mtx = utils.create_adjacency_matrix(decks)

    np.save(data / 'adj_mtx.npy', adj_mtx)

    with open(maps / 'int_to_card.json', 'w') as out_lookup:
        json.dump(int_to_card,  out_lookup)
